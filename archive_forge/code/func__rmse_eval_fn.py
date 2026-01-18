import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _rmse_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(targets, predictions, squared=False, sample_weight=sample_weight)
        return MetricValue(aggregate_results={'root_mean_squared_error': rmse})