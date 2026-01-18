from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class UnicodeIteratorType(SimpleIteratorType):

    def __init__(self, dtype):
        name = 'iter_unicode'
        self.data = dtype
        super(UnicodeIteratorType, self).__init__(name, dtype)