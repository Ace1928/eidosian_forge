from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class UnableToResolveVariableException(Exception):
    pass