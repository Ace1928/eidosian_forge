import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class BadIndexValue(errors.BzrError):
    _fmt = "The value '%(value)s' is not a valid value."

    def __init__(self, value):
        errors.BzrError.__init__(self)
        self.value = value