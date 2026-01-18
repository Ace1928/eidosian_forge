import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def _wrap_data_with_container(method, data_to_wrap, original_input, estimator):
    """Wrap output with container based on an estimator's or global config.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method to get container output for.

    data_to_wrap : {ndarray, dataframe}
        Data to wrap with container.

    original_input : {ndarray, dataframe}
        Original input of function.

    estimator : estimator instance
        Estimator with to get the output configuration from.

    Returns
    -------
    output : {ndarray, dataframe}
        If the output config is "default" or the estimator is not configured
        for wrapping return `data_to_wrap` unchanged.
        If the output config is "pandas", return `data_to_wrap` as a pandas
        DataFrame.
    """
    output_config = _get_output_config(method, estimator)
    if output_config['dense'] == 'default' or not _auto_wrap_is_configured(estimator):
        return data_to_wrap
    dense_config = output_config['dense']
    if issparse(data_to_wrap):
        raise ValueError(f"The transformer outputs a scipy sparse matrix. Try to set the transformer output to a dense array or disable {dense_config.capitalize()} output with set_output(transform='default').")
    adapter = ADAPTERS_MANAGER.adapters[dense_config]
    return adapter.create_container(data_to_wrap, original_input, columns=estimator.get_feature_names_out)