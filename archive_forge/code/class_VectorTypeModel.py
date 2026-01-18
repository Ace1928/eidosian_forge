from typing import List, Tuple, Dict
from numba import types
from numba.core import cgutils
from numba.core.extending import make_attribute_wrapper, models, register_model
from numba.core.imputils import Registry as ImplRegistry
from numba.core.typing.templates import ConcreteTemplate
from numba.core.typing.templates import Registry as TypingRegistry
from numba.core.typing.templates import signature
from numba.cuda import stubs
from numba.cuda.errors import CudaLoweringError
class VectorTypeModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [(attr_name, base_type) for attr_name in attr_names]
        super().__init__(dmm, fe_type, members)