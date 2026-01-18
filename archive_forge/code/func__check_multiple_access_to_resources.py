import collections
import contextlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _check_multiple_access_to_resources(self, captured_resources, exclusive_resource_access):
    """Raise if captured_resources are accessed by another CriticalSection.

    Args:
      captured_resources: Set of tensors of type resource.
      exclusive_resource_access: Whether this execution requires exclusive
        resource access.

    Raises:
      ValueError: If any tensors in `captured_resources` are also accessed
        by another `CriticalSection`, and at least one of them requires
        exclusive resource access.
    """
    for sg in ops.get_collection(CRITICAL_SECTION_EXECUTIONS):
        if self._is_self_handle(sg.handle):
            continue
        if not (exclusive_resource_access or sg.exclusive_resource_access):
            continue
        resource_intersection = captured_resources.intersection(sg.resources)
        if resource_intersection:
            raise ValueError(f"This execution would access resources: {list(resource_intersection)}. Either this lock (CriticalSection: {self._handle}) or lock '{sg}' (CriticalSection: {sg.handle}) requested exclusive resource access of this resource. Did you mean to call execute with keyword argument exclusive_resource_access=False?")