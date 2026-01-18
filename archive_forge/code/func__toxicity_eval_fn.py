import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _toxicity_eval_fn(predictions, targets=None, metrics=None):
    if not _validate_text_data(predictions, 'toxicity', predictions_col_specifier):
        return
    try:
        toxicity = _cached_evaluate_load('toxicity', module_type='measurement')
    except Exception as e:
        _logger.warning(f"Failed to load 'toxicity' metric (error: {e!r}), skipping metric logging.")
        return
    scores = toxicity.compute(predictions=predictions)['toxicity']
    toxicity_ratio = toxicity.compute(predictions=predictions, aggregation='ratio')['toxicity_ratio']
    return MetricValue(scores=scores, aggregate_results={**standard_aggregations(scores), 'ratio': toxicity_ratio})