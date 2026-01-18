from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
@register_model(types.NumPyRandomBitGeneratorType)
class NumPyRngBitGeneratorModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('parent', types.pyobject), ('state_address', types.uintp), ('state', types.uintp), ('fnptr_next_uint64', types.uintp), ('fnptr_next_uint32', types.uintp), ('fnptr_next_double', types.uintp), ('bit_generator', types.uintp)]
        super(NumPyRngBitGeneratorModel, self).__init__(dmm, fe_type, members)