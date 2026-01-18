import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
class IntEnumClass(EnumClass):
    """
    Type class for IntEnum classes.
    """
    basename = 'IntEnum class'

    @cached_property
    def member_type(self):
        """
        The type of this class' members.
        """
        return IntEnumMember(self.instance_class, self.dtype)