from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class UnicodeType(IterableType, Hashable):

    def __init__(self, name):
        super(UnicodeType, self).__init__(name)

    @property
    def iterator_type(self):
        return UnicodeIteratorType(self)