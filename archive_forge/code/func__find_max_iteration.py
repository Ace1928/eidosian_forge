import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
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