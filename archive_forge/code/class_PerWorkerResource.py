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
class PerWorkerResource:
    """A per-worker CapturableResource class for non-ParameterServer strategy.

  Resources that populate `host_to_resources` should be instances of classes
  subclassing CapturableResource, although currently it's only used and tested
  for StaticHashTable with TPUStrategy.
  """

    def __init__(self, strategy, host_to_resources):
        distribute_lib.distribution_strategy_input_api_counter.get_cell('PerWorkerResource', 'TPUDistributedLookupTable').increase_by(1)
        self._strategy = strategy
        self._host_to_resources = host_to_resources

    def __getattribute__(self, name):
        if name not in ('__init__', '__getattribute__', '_host_to_resources', '_strategy', 'local_resource'):
            return getattr(self.local_resource(), name)
        return super(PerWorkerResource, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in ('_strategy', '_host_to_resources'):
            return setattr(self.local_resource(), name, value)
        return super(PerWorkerResource, self).__setattr__(name, value)

    def local_resource(self):
        """Returns the resource on the local worker."""
        current_device = device_util.canonicalize(device_util.current())
        host_device = device_util.canonicalize(device_util.get_host_for_device(current_device))
        return self._host_to_resources.get(host_device, self._host_to_resources[next(iter(self._host_to_resources))])