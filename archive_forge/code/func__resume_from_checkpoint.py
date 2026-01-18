import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
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