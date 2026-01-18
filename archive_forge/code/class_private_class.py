from ._parser import parse, parser, parserinfo, ParserError
from ._parser import DEFAULTPARSER, DEFAULTTZPARSER
from ._parser import UnknownTimezoneWarning
from ._parser import __doc__
from .isoparser import isoparser, isoparse
from ._parser import _timelex, _resultbase
from ._parser import _tzparser, _parsetz
class private_class(c):
    __doc__ = c.__doc__

    def __init__(self, *args, **kwargs):
        warnings.warn(msg, DeprecationWarning)
        super(private_class, self).__init__(*args, **kwargs)