from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _finalize_scipy(module, available):
    if available:
        import scipy.stats
        import scipy.integrate
        import scipy.sparse
        import scipy.spatial