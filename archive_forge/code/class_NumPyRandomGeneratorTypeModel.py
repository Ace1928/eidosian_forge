from llvmlite import ir
from numba.core import cgutils, types
from numba.core.extending import (intrinsic, make_attribute_wrapper, models,
from numba import float32
@register_model(types.NumPyRandomGeneratorType)
class NumPyRandomGeneratorTypeModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('bit_generator', _bit_gen_type), ('meminfo', types.MemInfoPointer(types.voidptr)), ('parent', types.pyobject)]
        super(NumPyRandomGeneratorTypeModel, self).__init__(dmm, fe_type, members)