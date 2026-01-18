import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _validate_text_data(data, metric_name, col_specifier):
    """Validates that the data is a list of strs and is non-empty"""
    if data is None or len(data) == 0:
        _logger.warning(f'Cannot calculate {metric_name} for empty inputs: {col_specifier} is empty or the parameter is not specified. Skipping metric logging.')
        return False
    for row, line in enumerate(data):
        if not isinstance(line, str):
            _logger.warning(f'Cannot calculate {metric_name} for non-string inputs. Non-string found for {col_specifier} on row {row}. Skipping metric logging.')
            return False
    return True