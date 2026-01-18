import functools
import warnings
from functools import update_wrapper
import joblib
from .._config import config_context, get_config
def _with_config(delayed_func, config):
    """Helper function that intends to attach a config to a delayed function."""
    if hasattr(delayed_func, 'with_config'):
        return delayed_func.with_config(config)
    else:
        warnings.warn('`sklearn.utils.parallel.Parallel` needs to be used in conjunction with `sklearn.utils.parallel.delayed` instead of `joblib.delayed` to correctly propagate the scikit-learn configuration to the joblib workers.', UserWarning)
        return delayed_func