from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class DjangoFormResolver(DefaultResolver):

    def get_dictionary(self, var, names=None):
        names, used___dict__ = self.get_names(var)
        has_errors_attr = False
        if 'errors' in names:
            has_errors_attr = True
            names.remove('errors')
        d = defaultResolver.get_dictionary(var, names=names, used___dict__=used___dict__)
        if has_errors_attr:
            try:
                errors_attr = getattr(var, '_errors')
            except:
                errors_attr = None
            d['errors'] = errors_attr
        return d