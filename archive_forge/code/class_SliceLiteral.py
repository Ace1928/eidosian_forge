from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class SliceLiteral(Literal, SliceType):

    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[slice]({})'.format(value)
        members = 2 if value.step is None else 3
        SliceType.__init__(self, name=name, members=members)

    @property
    def key(self):
        sl = self.literal_value
        return (sl.start, sl.stop, sl.step)