import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def _create_placeholder_helper(self, graph: Any, tensor: core.Tensor, name: str):
    """A helper function to create capture placeholder."""
    placeholder = self._by_val_internal.get(id(tensor))
    if placeholder is None:
        tracing_ctx = trace_type.InternalTracingContext()
        spec = trace_type.from_value(tensor, tracing_ctx)
        spec._name = name
        if isinstance(tensor, core.Value) and tensor.is_packed:
            composite_device_name = tensor.device
        else:
            composite_device_name = None
        placeholder_ctx = trace_type.InternalPlaceholderContext(graph, with_none_control_dependencies=True, composite_device_name=composite_device_name)
        placeholder = spec.placeholder_value(placeholder_ctx)
        self.add_or_replace(key=id(tensor), external=tensor, internal=placeholder, is_by_ref=False)
        graph.inputs.append(placeholder)
    placeholder._record_tape(tensor)
    return placeholder