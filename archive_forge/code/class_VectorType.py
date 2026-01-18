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
class VectorType(types.Type):

    def __init__(self, name, base_type, attr_names, user_facing_object):
        self._base_type = base_type
        self._attr_names = attr_names
        self._user_facing_object = user_facing_object
        super().__init__(name=name)

    @property
    def base_type(self):
        return self._base_type

    @property
    def attr_names(self):
        return self._attr_names

    @property
    def num_elements(self):
        return len(self._attr_names)

    @property
    def user_facing_object(self):
        return self._user_facing_object