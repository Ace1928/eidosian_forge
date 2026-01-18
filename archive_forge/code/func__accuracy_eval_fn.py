import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _accuracy_eval_fn(predictions, targets=None, metrics=None, sample_weight=None):
    if targets is not None and len(targets) != 0:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true=targets, y_pred=predictions, sample_weight=sample_weight)
        return MetricValue(aggregate_results={'exact_match': acc})