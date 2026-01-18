from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _workaround_for_static_import_finders():
    import pycparser.yacctab
    import pycparser.lextab