import copy
import logging
import sys
import warnings
from .event_handler import MetricHandler, ValidationHandler, LoggingHandler, StoppingHandler, GradientUpdateHandler
from .event_handler import TrainBegin, EpochBegin, BatchBegin, BatchEnd, EpochEnd, TrainEnd
from .event_handler import _check_event_handlers
from .utils import _check_metrics, _suggest_metric_for_loss, _check_handler_metric_ref
from ...data import DataLoader
from ...loss import Loss as gluon_loss
from ...trainer import Trainer
from ...utils import split_and_load
from ....context import Context, cpu, gpu, num_gpus
from ....metric import Loss as metric_loss
from .batch_processor import BatchProcessor
def _check_batch_processor(self, batch_processor):
    if batch_processor is not None:
        model_fit = getattr(batch_processor, 'fit_batch', None)
        model_evaluate = getattr(batch_processor, 'evaluate_batch', None)
        if not callable(model_fit) or not callable(model_evaluate):
            raise ValueError('Customized Batch Processor must contain fit_batch() and evaluate_batch() methods')
    else:
        batch_processor = BatchProcessor()
    return batch_processor