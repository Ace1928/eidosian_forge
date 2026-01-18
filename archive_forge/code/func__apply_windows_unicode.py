import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _apply_windows_unicode(self, kwds):
    defmacros = kwds.get('define_macros', ())
    if not isinstance(defmacros, (list, tuple)):
        raise TypeError("'define_macros' must be a list or tuple")
    defmacros = list(defmacros) + [('UNICODE', '1'), ('_UNICODE', '1')]
    kwds['define_macros'] = defmacros