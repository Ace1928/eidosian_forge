import operator
import warnings
from llvmlite import ir
from numba.core import types, cgutils
from numba.core import typing
from numba.core.registry import cpu_target
from numba.core.typeconv import Conversion
from numba.core.extending import intrinsic
from numba.core.errors import TypingError, NumbaTypeSafetyWarning
def _container_get_data(context, builder, container_ty, c):
    """Helper to get the C list pointer in a numba containers.
    """
    ctor = cgutils.create_struct_proxy(container_ty)
    conatainer_struct = ctor(context, builder, value=c)
    return conatainer_struct.data