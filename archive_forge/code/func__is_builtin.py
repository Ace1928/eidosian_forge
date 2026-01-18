from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
def _is_builtin(obj):
    """Check if an object need not be converted."""
    return isinstance(obj, (float, int, str, bool))