import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _ari_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(predictions, 'ari', predictions_col_specifier):
        return
    try:
        import textstat
    except ImportError:
        _logger.warning("Failed to import textstat for automated readability index metric, skipping metric logging. Please install textstat using 'pip install textstat'.")
        return
    scores = [textstat.automated_readability_index(prediction) for prediction in predictions]
    return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))