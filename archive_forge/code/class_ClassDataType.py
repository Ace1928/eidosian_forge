from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
class ClassDataType(Type):
    """
    Internal only.
    Represents the data of the instance.  The representation of
    ClassInstanceType contains a pointer to a ClassDataType which represents
    a C structure that contains all the data fields of the class instance.
    """

    def __init__(self, classtyp):
        self.class_type = classtyp
        name = 'data.{0}'.format(self.class_type.name)
        super(ClassDataType, self).__init__(name)