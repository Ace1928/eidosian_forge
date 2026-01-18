import functools
from llvmlite import ir
from numba.core.datamodel.registry import DataModelManager, register
from numba.core.extending import models
from numba.core import types
from numba.cuda.types import Dim3, GridGroup, CUDADispatcher
@register_model(GridGroup)
class GridGroupModel(models.PrimitiveModel):

    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(64)
        super().__init__(dmm, fe_type, be_type)