from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class ExceptionInstance(Phantom):
    """
    The type of exception instances.  *exc_class* should be the
    exception class.
    """

    def __init__(self, exc_class):
        assert issubclass(exc_class, BaseException)
        name = '%s(...)' % (exc_class.__name__,)
        self.exc_class = exc_class
        super(ExceptionInstance, self).__init__(name)

    @property
    def key(self):
        return self.exc_class