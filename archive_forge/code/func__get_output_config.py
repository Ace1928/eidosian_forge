import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def _get_output_config(method, estimator=None):
    """Get output config based on estimator and global configuration.

    Parameters
    ----------
    method : {"transform"}
        Estimator's method for which the output container is looked up.

    estimator : estimator instance or None
        Estimator to get the output configuration from. If `None`, check global
        configuration is used.

    Returns
    -------
    config : dict
        Dictionary with keys:

        - "dense": specifies the dense container for `method`. This can be
          `"default"` or `"pandas"`.
    """
    est_sklearn_output_config = getattr(estimator, '_sklearn_output_config', {})
    if method in est_sklearn_output_config:
        dense_config = est_sklearn_output_config[method]
    else:
        dense_config = get_config()[f'{method}_output']
    supported_outputs = ADAPTERS_MANAGER.supported_outputs
    if dense_config not in supported_outputs:
        raise ValueError(f'output config must be in {sorted(supported_outputs)}, got {dense_config}')
    return {'dense': dense_config}