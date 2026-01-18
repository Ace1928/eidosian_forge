import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class ValidationHandler(TrainBegin, BatchEnd, EpochEnd):
    """Validation Handler that evaluate model on validation dataset

    :py:class:`ValidationHandler` takes validation dataset, an evaluation function,
    metrics to be evaluated, and how often to run the validation. You can provide custom
    evaluation function or use the one provided my :py:class:`Estimator`

    Parameters
    ----------
    val_data : DataLoader
        Validation data set to run evaluation.
    eval_fn : function
        A function defines how to run evaluation and
        calculate loss and metrics.
    epoch_period : int, default 1
        How often to run validation at epoch end, by default
        :py:class:`ValidationHandler` validate every epoch.
    batch_period : int, default None
        How often to run validation at batch end, by default
        :py:class:`ValidationHandler` does not validate at batch end.
    priority: scalar, default -1000
        Priority level of the ValidationHandler. Priority level is sorted in
        ascending order. The lower the number is, the higher priority level the
        handler is.
    event_handlers : EventHandler or list of EventHandlers
        List of :py:class:`EventHandler` to apply during validaiton. This argument
        is used by self.eval_fn function in order to process customized event
        handlers.
    """

    def __init__(self, val_data, eval_fn, epoch_period=1, batch_period=None, priority=-1000, event_handlers=None):
        self.val_data = val_data
        self.eval_fn = eval_fn
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.current_batch = 0
        self.current_epoch = 0
        self.priority = priority
        self.event_handlers = event_handlers

    def train_begin(self, estimator, *args, **kwargs):
        self.current_batch = 0
        self.current_epoch = 0

    def batch_end(self, estimator, *args, **kwargs):
        self.current_batch += 1
        if self.batch_period and self.current_batch % self.batch_period == 0:
            self.eval_fn(val_data=self.val_data, batch_axis=estimator.batch_axis, event_handlers=self.event_handlers)

    def epoch_end(self, estimator, *args, **kwargs):
        self.current_epoch += 1
        if self.epoch_period and self.current_epoch % self.epoch_period == 0:
            self.eval_fn(val_data=self.val_data, batch_axis=estimator.batch_axis, event_handlers=self.event_handlers)