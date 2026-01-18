from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class StructRefPayload(Type):
    """The type of the payload of a mutable struct.
    """
    mutable = True

    def __init__(self, typename, fields):
        self._typename = typename
        self._fields = tuple(fields)
        super().__init__(name=f'numba.{typename}{self._fields}.payload')

    @property
    def field_dict(self):
        return MappingProxyType(dict(self._fields))