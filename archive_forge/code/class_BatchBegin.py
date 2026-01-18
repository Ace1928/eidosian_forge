import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class BatchBegin(EventHandler):

    def batch_begin(self, estimator, *args, **kwargs):
        pass