import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _check_array_api_dispatch(array_api_dispatch):
    """Check that array_api_compat is installed and NumPy version is compatible.

    array_api_compat follows NEP29, which has a higher minimum NumPy version than
    scikit-learn.
    """
    if array_api_dispatch:
        try:
            import array_api_compat
        except ImportError:
            raise ImportError('array_api_compat is required to dispatch arrays using the API specification')
        numpy_version = parse_version(numpy.__version__)
        min_numpy_version = '1.21'
        if numpy_version < parse_version(min_numpy_version):
            raise ImportError(f'NumPy must be {min_numpy_version} or newer to dispatch array using the API specification')