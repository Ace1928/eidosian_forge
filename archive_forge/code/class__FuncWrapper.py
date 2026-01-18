import functools
import warnings
from functools import update_wrapper
import joblib
from .._config import config_context, get_config
class _FuncWrapper:
    """Load the global configuration before calling the function."""

    def __init__(self, function):
        self.function = function
        update_wrapper(self, self.function)

    def with_config(self, config):
        self.config = config
        return self

    def __call__(self, *args, **kwargs):
        config = getattr(self, 'config', None)
        if config is None:
            warnings.warn('`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.', UserWarning)
            config = {}
        with config_context(**config):
            return self.function(*args, **kwargs)