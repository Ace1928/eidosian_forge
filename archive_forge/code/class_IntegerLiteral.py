import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
class IntegerLiteral(Literal, Integer):

    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[int]({})'.format(value)
        basetype = self.literal_type
        Integer.__init__(self, name=name, bitwidth=basetype.bitwidth, signed=basetype.signed)

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)