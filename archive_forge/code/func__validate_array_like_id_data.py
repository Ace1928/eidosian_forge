import functools
import logging
import os
import numpy as np
from mlflow.metrics.base import MetricValue, standard_aggregations
def _validate_array_like_id_data(data, metric_name, col_specifier):
    """Validates that the data is a list of lists/np.ndarrays of strings/ints and is non-empty"""
    if data is None or len(data) == 0:
        return False
    for index, value in data.items():
        if not (isinstance(value, list) and all((isinstance(val, (str, int)) for val in value)) or (isinstance(value, np.ndarray) and (np.issubdtype(value.dtype, str) or np.issubdtype(value.dtype, int)))):
            _logger.warning(f"Cannot calculate metric '{metric_name}' for non-arraylike of string or int inputs. Non-arraylike of strings/ints found for {col_specifier} on row {index}, value {value}. Skipping metric logging.")
            return False
    return True