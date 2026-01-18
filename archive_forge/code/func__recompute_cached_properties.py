import collections as py_collections
import functools
from typing import Any, Callable, Hashable, Mapping, Optional
from tensorflow.core.function import trace_type
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
def _recompute_cached_properties(self):
    """Regenerates cached properties if there have been mutations."""
    self._by_val_internal.mutated = False
    self._by_val_external.mutated = False
    assert len(self._by_val_internal) == len(self._by_val_external)
    self._cached_by_val_capture_tuples = []
    for key in self._by_val_internal:
        assert key in self._by_val_external
        internal = self._by_val_internal[key]
        external = self._by_val_external[key]
        self._cached_by_val_capture_tuples.append((external, internal))
    self._cached_capture_types = py_collections.OrderedDict(list(self._by_val_tracetype.items()) + list(self._by_ref_tracetype.items()))