import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _recall_at_k_eval_fn(k):
    if not (isinstance(k, int) and k > 0):
        _logger.warning(f"Cannot calculate 'precision_at_k' for invalid parameter 'k'. 'k' should be a positive integer; found: {k}. Skipping metric logging.")
        return noop

    def _fn(predictions, targets):
        if not _validate_array_like_id_data(predictions, 'precision_at_k', predictions_col_specifier) or not _validate_array_like_id_data(targets, 'precision_at_k', targets_col_specifier):
            return
        scores = []
        for target, prediction in zip(targets, predictions):
            ground_truth, retrieved = (set(target), set(prediction[:k]))
            relevant_doc_count = len(ground_truth.intersection(retrieved))
            if len(ground_truth) > 0:
                scores.append(relevant_doc_count / len(ground_truth))
            elif len(retrieved) == 0:
                scores.append(1)
            else:
                scores.append(0)
        return MetricValue(scores=scores, aggregate_results=standard_aggregations(scores))
    return _fn