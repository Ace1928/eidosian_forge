import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class BadIndexFormatSignature(errors.BzrError):
    _fmt = '%(value)s is not an index of type %(_type)s.'

    def __init__(self, value, _type):
        errors.BzrError.__init__(self)
        self.value = value
        self._type = _type