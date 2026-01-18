import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def _capture_by_ref(self, graph: Any, lam: Callable[[], Any], key: Hashable=None) -> Any:
    """Used during tracing process to create/retrive by-ref captures.

    Args:
      graph: The FuncGraph that captures this tensor.
      lam: A callable that takes no arguments and returns tensor captures.
      key: A hashable identifier.

    Returns:
      Tensor from this FuncGraph.
    """
    if key is not None and key in self._by_ref_internal:
        return self._by_ref_internal[key]
    if key is None:
        key = len(self._by_ref_internal)
        while key in self._by_ref_internal:
            key += 1
    value_nested = lam()
    capture_trace_type = trace_type.from_value(value_nested)
    ctx = trace_type.InternalPlaceholderContext(graph)
    internal = capture_trace_type.placeholder_value(ctx)

    def lam_fn():
        value = lam()
        return capture_trace_type._to_tensors(value)
    self._by_ref_external[key] = lam_fn
    self._by_ref_internal[key] = internal
    self._by_ref_tracetype[key] = capture_trace_type
    return self._by_ref_internal[key]