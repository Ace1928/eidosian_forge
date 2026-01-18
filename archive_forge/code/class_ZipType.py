from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.ZipType)
class ZipType(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('iter%d' % i, source_type.iterator_type) for i, source_type in enumerate(fe_type.source_types)]
        super(ZipType, self).__init__(dmm, fe_type, members)