from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
def _console_warning(s, log_fn):
    s = s.strip()
    if s == '=':
        return
    else:
        return log_fn(rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG, s)