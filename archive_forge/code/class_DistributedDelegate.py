import copy
from typing import Optional
import weakref
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace
class DistributedDelegate(DistributedValues):
    """A map from device to values; acts as the same type as the values."""

    def __getattr__(self, name):
        if name.startswith('_self_') or name in ('_use_resource_variables', '_attribute_sentinel', '_distributed_container'):
            return super(DistributedDelegate, self).__getattr__(name)
        if name == '_values':
            raise AttributeError()
        return getattr(self._get(), name)

    @property
    def values(self):
        """Returns the per replica values."""
        return self._values

    def _get_as_operand(self):
        """Returns the value for operations for the current device.

    Some implementations, e.g. `TPUMirroredVariable`, are not able to return the
    value type within a replica context. They can, however, return a value that
    can be used by the operations below.
    """
        return self._get()

    def __add__(self, o):
        return self._get_as_operand() + o

    def __radd__(self, o):
        return o + self._get_as_operand()

    def __sub__(self, o):
        return self._get_as_operand() - o

    def __rsub__(self, o):
        return o - self._get_as_operand()

    def __mul__(self, o):
        return self._get_as_operand() * o

    def __rmul__(self, o):
        return o * self._get_as_operand()

    def __truediv__(self, o):
        return self._get_as_operand() / o

    def __rtruediv__(self, o):
        return o / self._get_as_operand()

    def __floordiv__(self, o):
        return self._get_as_operand() // o

    def __rfloordiv__(self, o):
        return o // self._get_as_operand()

    def __mod__(self, o):
        return self._get_as_operand() % o

    def __rmod__(self, o):
        return o % self._get_as_operand()

    def __lt__(self, o):
        return self._get_as_operand() < o

    def __le__(self, o):
        return self._get_as_operand() <= o

    def __gt__(self, o):
        return self._get_as_operand() > o

    def __ge__(self, o):
        return self._get_as_operand() >= o

    def __and__(self, o):
        return self._get_as_operand() & o

    def __rand__(self, o):
        return o & self._get_as_operand()

    def __or__(self, o):
        return self._get_as_operand() | o

    def __ror__(self, o):
        return o | self._get_as_operand()

    def __xor__(self, o):
        return self._get_as_operand() ^ o

    def __rxor__(self, o):
        return o ^ self._get_as_operand()

    def __getitem__(self, o):
        return self._get_as_operand()[o]

    def __pow__(self, o, modulo=None):
        return pow(self._get_as_operand(), o, modulo)

    def __rpow__(self, o):
        return pow(o, self._get_as_operand())

    def __invert__(self):
        return ~self._get_as_operand()

    def __neg__(self):
        return -self._get_as_operand()

    def __abs__(self):
        return abs(self._get_as_operand())

    def __div__(self, o):
        try:
            return self._get_as_operand().__div__(o)
        except AttributeError:
            return NotImplemented

    def __rdiv__(self, o):
        try:
            return self._get_as_operand().__rdiv__(o)
        except AttributeError:
            return NotImplemented

    def __matmul__(self, o):
        try:
            return self._get_as_operand().__matmul__(o)
        except AttributeError:
            return NotImplemented

    def __rmatmul__(self, o):
        try:
            return self._get_as_operand().__rmatmul__(o)
        except AttributeError:
            return NotImplemented