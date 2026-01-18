import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class LoggingHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):
    """Basic Logging Handler that applies to every Gluon estimator by default.

    :py:class:`LoggingHandler` logs hyper-parameters, training statistics,
    and other useful information during training

    Parameters
    ----------
    log_interval: int or str, default 'epoch'
        Logging interval during training.
        log_interval='epoch': display metrics every epoch
        log_interval=integer k: display metrics every interval of k batches
    metrics : list of EvalMetrics
        Metrics to be logged, logged at batch end, epoch end, train end.
    priority : scalar, default np.Inf
        Priority level of the LoggingHandler. Priority level is sorted in
        ascending order. The lower the number is, the higher priority level the
        handler is.
    """

    def __init__(self, log_interval='epoch', metrics=None, priority=np.Inf):
        super(LoggingHandler, self).__init__()
        if not isinstance(log_interval, int) and log_interval != 'epoch':
            raise ValueError("log_interval must be either an integer or string 'epoch'")
        self.metrics = _check_metrics(metrics)
        self.batch_index = 0
        self.current_epoch = 0
        self.processed_samples = 0
        self.priority = priority
        self.log_interval = log_interval
        self.log_interval_time = 0

    def train_begin(self, estimator, *args, **kwargs):
        self.train_start = time.time()
        trainer = estimator.trainer
        optimizer = trainer.optimizer.__class__.__name__
        lr = trainer.learning_rate
        estimator.logger.info('Training begin: using optimizer %s with current learning rate %.4f ', optimizer, lr)
        if estimator.max_epoch:
            estimator.logger.info('Train for %d epochs.', estimator.max_epoch)
        else:
            estimator.logger.info('Train for %d batches.', estimator.max_batch)
        self.current_epoch = 0
        self.batch_index = 0
        self.processed_samples = 0
        self.log_interval_time = 0

    def train_end(self, estimator, *args, **kwargs):
        train_time = time.time() - self.train_start
        msg = 'Train finished using total %ds with %d epochs. ' % (train_time, self.current_epoch)
        for metric in self.metrics:
            name, value = metric.get()
            msg += '%s: %.4f, ' % (name, value)
        estimator.logger.info(msg.rstrip(', '))

    def batch_begin(self, estimator, *args, **kwargs):
        if isinstance(self.log_interval, int):
            self.batch_start = time.time()

    def batch_end(self, estimator, *args, **kwargs):
        if isinstance(self.log_interval, int):
            batch_time = time.time() - self.batch_start
            msg = '[Epoch %d][Batch %d]' % (self.current_epoch, self.batch_index)
            self.processed_samples += kwargs['batch'][0].shape[0]
            msg += '[Samples %s] ' % self.processed_samples
            self.log_interval_time += batch_time
            if self.batch_index % self.log_interval == 0:
                msg += 'time/interval: %.3fs ' % self.log_interval_time
                self.log_interval_time = 0
                for metric in self.metrics:
                    name, value = metric.get()
                    msg += '%s: %.4f, ' % (name, value)
                estimator.logger.info(msg.rstrip(', '))
        self.batch_index += 1

    def epoch_begin(self, estimator, *args, **kwargs):
        if isinstance(self.log_interval, int) or self.log_interval == 'epoch':
            is_training = False
            for metric in self.metrics:
                if 'training' in metric.name:
                    is_training = True
            self.epoch_start = time.time()
            if is_training:
                estimator.logger.info('[Epoch %d] Begin, current learning rate: %.4f', self.current_epoch, estimator.trainer.learning_rate)
            else:
                estimator.logger.info('Validation Begin')

    def epoch_end(self, estimator, *args, **kwargs):
        if isinstance(self.log_interval, int) or self.log_interval == 'epoch':
            epoch_time = time.time() - self.epoch_start
            msg = '[Epoch %d] Finished in %.3fs, ' % (self.current_epoch, epoch_time)
            for monitor in self.metrics:
                name, value = monitor.get()
                msg += '%s: %.4f, ' % (name, value)
            estimator.logger.info(msg.rstrip(', '))
        self.current_epoch += 1
        self.batch_index = 0