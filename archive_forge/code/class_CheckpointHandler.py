import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class CheckpointHandler(TrainBegin, BatchEnd, EpochEnd):
    """Save the model after user define period

    :py:class:`CheckpointHandler` saves the network architecture after first batch if the model
    can be fully hybridized, saves model parameters and trainer states after user defined period,
    default saves every epoch.

    Parameters
    ----------
    model_dir : str
        File directory to save all the model related files including model architecture,
        model parameters, and trainer states.
    model_prefix : str default 'model'
        Prefix to add for all checkpoint file names.
    monitor: EvalMetric, default None
        The metrics to monitor and determine if model has improved
    verbose: int, default 0
        Verbosity mode, 1 means inform user every time a checkpoint is saved
    save_best: bool, default False
        If True, monitor must not be None, :py:class:`CheckpointHandler` will save the
        model parameters and trainer states with the best monitored value.
    mode: str, default 'auto'
        One of {auto, min, max}, if `save_best=True`, the comparison to make
        and determine if the monitored value has improved. if 'auto' mode,
        :py:class:`CheckpointHandler` will try to use min or max based on
        the monitored metric name.
    epoch_period: int, default 1
        Epoch intervals between saving the network. By default, checkpoints are
        saved every epoch.
    batch_period: int, default None
        Batch intervals between saving the network.
        By default, checkpoints are not saved based on the number of batches.
    max_checkpoints : int, default 5
        Maximum number of checkpoint files to keep in the model_dir, older checkpoints
        will be removed. Best checkpoint file is not counted.
    resume_from_checkpoint : bool, default False
        Whether to resume training from checkpoint in model_dir. If True and checkpoints
        found, :py:class:`CheckpointHandler` will load net parameters and trainer states,
        and train the remaining of epochs and batches.
    """

    def __init__(self, model_dir, model_prefix='model', monitor=None, verbose=0, save_best=False, mode='auto', epoch_period=1, batch_period=None, max_checkpoints=5, resume_from_checkpoint=False):
        self.monitor = monitor
        self.verbose = verbose
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self.save_best = save_best
        if self.save_best and (not isinstance(self.monitor, EvalMetric)):
            raise ValueError('To save best model only, please provide one of the metric objects from estimator.train_metrics and estimator.val_metrics as monitor.')
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.current_batch = 0
        self.current_epoch = 0
        self.max_checkpoints = max_checkpoints
        self.resume_from_checkpoint = resume_from_checkpoint
        self.saved_checkpoints = []
        if self.save_best:
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode. CheckpointHandler will usemax mode for f1 and accuracy metric comparison and use min mode other wise' % mode, RuntimeWarning)
                mode = 'auto'
            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            elif 'acc' or 'f1' in self.monitor.get()[0].lower():
                warnings.warn("`greater` operator will be used to determine if {} has improved. Please specify `mode='min'` to use the `less` operator. Specify `mode='max' to disable this warning.`".format(self.monitor.get()[0]))
                self.monitor_op = np.greater
            else:
                warnings.warn("`less` operator will be used to determine if {} has improved. Please specify `mode='max'` to use the `greater` operator. Specify `mode='min' to disable this warning.`".format(self.monitor.get()[0]))
                self.monitor_op = np.less

    def train_begin(self, estimator, *args, **kwargs):
        self.current_epoch = 0
        self.current_batch = 0
        if self.save_best:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        if self.resume_from_checkpoint:
            error_msg = 'To use resume from checkpoint, you must only specify the same type of period you used for training.For example, if you are training based on number of epochs,you must save only based on epochs, and set batch_period to None.'
            if estimator.max_batch:
                assert self.batch_period, error_msg
                assert not self.epoch_period, error_msg
            if estimator.max_epoch:
                assert self.epoch_period, error_msg
                assert not self.batch_period, error_msg
            self._resume_from_checkpoint(estimator)

    def batch_end(self, estimator, *args, **kwargs):
        if self.current_batch == 0:
            self._save_symbol(estimator)
        if self.batch_period and (self.current_batch + 1) % self.batch_period == 0:
            self._save_checkpoint(estimator)
        self.current_batch += 1

    def epoch_end(self, estimator, *args, **kwargs):
        if self.epoch_period and (self.current_epoch + 1) % self.epoch_period == 0:
            self._save_checkpoint(estimator)
        self.current_epoch += 1

    def _save_checkpoint(self, estimator):
        if self.resume_from_checkpoint:
            save_epoch_number = self.current_epoch + self.trained_epoch + 1
            if estimator.max_epoch:
                save_batch_number = self.current_batch + self.trained_batch
            else:
                save_batch_number = self.current_batch + self.trained_batch + 1
        else:
            save_epoch_number = self.current_epoch
            save_batch_number = self.current_batch
        prefix = '%s-epoch%dbatch%d' % (self.model_prefix, save_epoch_number, save_batch_number)
        self._save_params_and_trainer(estimator, prefix)
        if self.verbose > 0:
            estimator.logger.info('[Epoch %d] CheckpointHandler: trained total %d batches, saving model at %s with prefix: %s', self.current_epoch, self.current_batch + 1, self.model_dir, prefix)
        if self.save_best:
            monitor_name, monitor_value = self.monitor.get()
            if np.isnan(monitor_value):
                warnings.warn(RuntimeWarning('Skipping save best because %s is not updated, make sure you pass one of the metric objects estimator.train_metrics and estimator.val_metrics as monitor', monitor_name))
            elif self.monitor_op(monitor_value, self.best):
                prefix = self.model_prefix + '-best'
                self._save_params_and_trainer(estimator, prefix)
                self.best = monitor_value
                if self.verbose > 0:
                    estimator.logger.info('[Epoch %d] CheckpointHandler: %s improved from %0.5f to %0.5f, updating best model at %s with prefix: %s', self.current_epoch, monitor_name, self.best, monitor_value, self.model_dir, prefix)
            elif self.verbose > 0:
                estimator.logger.info('[Epoch %d] CheckpointHandler: %s did not improve from %0.5f, skipping updating best model', self.current_batch, monitor_name, self.best)

    def _save_symbol(self, estimator):
        symbol_file = os.path.join(self.model_dir, self.model_prefix + '-symbol.json')
        if hasattr(estimator.net, '_cached_graph') and estimator.net._cached_graph:
            sym = estimator.net._cached_graph[1]
            sym.save(symbol_file)
        else:
            estimator.logger.info('Model architecture(symbol file) is not saved, please use HybridBlock to construct your model, and call net.hybridize() before passing to Estimator in order to save model architecture as %s.', symbol_file)

    def _save_params_and_trainer(self, estimator, file_prefix):
        param_file = os.path.join(self.model_dir, file_prefix + '.params')
        trainer_file = os.path.join(self.model_dir, file_prefix + '.states')
        estimator.net.save_parameters(param_file)
        estimator.trainer.save_states(trainer_file)
        if 'best' not in file_prefix:
            self.saved_checkpoints.append(file_prefix)
        if len(self.saved_checkpoints) > self.max_checkpoints:
            prefix = self.saved_checkpoints.pop(0)
            for fname in os.listdir(self.model_dir):
                if fname.startswith(prefix):
                    os.remove(os.path.join(self.model_dir, fname))

    def _resume_from_checkpoint(self, estimator):
        prefix = self.model_prefix + '-epoch'
        self.trained_epoch = self._find_max_iteration(dir=self.model_dir, prefix=prefix, start='epoch', end='batch', saved_checkpoints=self.saved_checkpoints)
        prefix += str(self.trained_epoch)
        self.trained_batch = self._find_max_iteration(dir=self.model_dir, prefix=prefix, start='batch', end='.params')
        if self.trained_epoch == -1:
            msg = 'CheckpointHandler: No checkpoint found, training from scratch for '
            if estimator.max_batch:
                msg += '%d batches' % estimator.max_batch
            else:
                msg += '%d epochs' % estimator.max_epoch
            estimator.logger.info(msg)
        else:
            msg = 'CheckpointHandler: Checkpoint resumed from epoch %d batch %d, continue to train for ' % (self.trained_epoch, self.trained_batch)
            if estimator.max_epoch:
                if self.trained_epoch >= estimator.max_epoch - 1:
                    raise ValueError('Found checkpoint with maximum number of epoch %d reached, please specify resume_from_checkpoint=False (default value) if you wan to train from scratch.' % estimator.max_epoch)
                estimator.max_epoch = estimator.max_epoch - self.trained_epoch - 1
                msg += '%d epochs ' % estimator.max_epoch
            if estimator.max_batch:
                if self.trained_batch >= estimator.max_batch - 1:
                    raise ValueError('Found checkpoint with maximum number of batch %d reached, please specifyresume_from_checkpoint=False (default value) if you wan to train from scratch.' % self.trained_batch)
                estimator.max_batch = estimator.max_batch - self.trained_batch - 1
                msg += '%d batches ' % estimator.max_batch
            param_file = '%s-epoch%dbatch%d.params' % (self.model_prefix, self.trained_epoch, self.trained_batch)
            param_file = os.path.join(self.model_dir, param_file)
            trainer_file = '%s-epoch%dbatch%d.states' % (self.model_prefix, self.trained_epoch, self.trained_batch)
            trainer_file = os.path.join(self.model_dir, trainer_file)
            assert os.path.exists(param_file), 'Failed to load checkpoint, %s does not exist' % param_file
            assert os.path.exists(trainer_file), 'Failed to load checkpoint, %s does not exist' % trainer_file
            estimator.net.load_parameters(param_file, ctx=estimator.context)
            estimator.trainer.load_states(trainer_file)
            estimator.logger.warning(msg)

    def _find_max_iteration(self, dir, prefix, start, end, saved_checkpoints=None):
        error_msg = 'Error parsing checkpoint file, please check your checkpoints have the format: {model_name}-epoch{epoch_number}batch{batch_number}.params, there should also be a .states file for each .params file '
        max_iter = -1
        for fname in os.listdir(dir):
            if fname.startswith(prefix) and '.params' in fname:
                if saved_checkpoints:
                    saved_checkpoints.append(fname[:fname.find('.params')])
                try:
                    iter = int(fname[fname.find(start) + len(start):fname.find(end)])
                    if iter > max_iter:
                        max_iter = iter
                except ValueError:
                    raise ValueError(error_msg)
        return max_iter