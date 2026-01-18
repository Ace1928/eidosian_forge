from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class ForwardInternalResolverToObject:
    """
    To be used when we provide some internal object that'll actually do the resolution.
    """

    def get_contents_debug_adapter_protocol(self, obj, fmt=None):
        return obj.get_contents_debug_adapter_protocol(fmt)

    def get_dictionary(self, var, fmt={}):
        return var.get_dictionary(var, fmt)

    def resolve(self, var, attribute):
        return var.resolve(attribute)