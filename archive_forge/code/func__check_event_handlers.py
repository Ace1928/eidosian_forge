import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
def _check_event_handlers(handlers):
    if isinstance(handlers, EventHandler):
        handlers = [handlers]
    else:
        handlers = handlers or []
        if not all([isinstance(handler, EventHandler) for handler in handlers]):
            raise ValueError('handlers must be an EventHandler or a list of EventHandler, got: {}'.format(handlers))
    return handlers