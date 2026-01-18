from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def _does_obj_repr_evaluate_to_obj(obj):
    """
    If obj is an object where evaluating its representation leads to
    the same object, return True, otherwise, return False.
    """
    try:
        if isinstance(obj, tuple):
            for o in obj:
                if not _does_obj_repr_evaluate_to_obj(o):
                    return False
            return True
        else:
            return isinstance(obj, _basic_immutable_types)
    except:
        return False