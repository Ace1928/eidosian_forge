from ._parser import parse, parser, parserinfo, ParserError
from ._parser import DEFAULTPARSER, DEFAULTTZPARSER
from ._parser import UnknownTimezoneWarning
from ._parser import __doc__
from .isoparser import isoparser, isoparse
from ._parser import _timelex, _resultbase
from ._parser import _tzparser, _parsetz
def __deprecated_private_func(f):
    from functools import wraps
    import warnings
    msg = '{name} is a private function and may break without warning, it will be moved and or renamed in future versions.'
    msg = msg.format(name=f.__name__)

    @wraps(f)
    def deprecated_func(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning)
        return f(*args, **kwargs)
    return deprecated_func