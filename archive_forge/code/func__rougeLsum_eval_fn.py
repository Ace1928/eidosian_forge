import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _rougeLsum_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(targets, 'rougeLsum', targets_col_specifier) or not _validate_text_data(predictions, 'rougeLsum', predictions_col_specifier):
        return
    try:
        rouge = _cached_evaluate_load('rouge')
    except Exception as e:
        _logger.warning(f"Failed to load 'rouge' metric (error: {e!r}), skipping metric logging.")
        return
    scores = rouge.compute(predictions=predictions, references=targets, rouge_types=['rougeLsum'], use_aggregator=False)['rougeLsum']
    return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))