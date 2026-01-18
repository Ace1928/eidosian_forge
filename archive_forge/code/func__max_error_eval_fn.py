import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _max_error_eval_fn(predictions, targets=None, metrics=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import max_error
        error = max_error(targets, predictions)
        return MetricValue(aggregate_results={'max_error': error})