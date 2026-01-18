import collections.abc
from typing import Any, Hashable, Optional, Dict
import weakref
from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class InternalTracingContext(trace.TracingContext):
    """Container for variables and flags shared across TraceType generation."""

    def __init__(self, is_legacy_signature: bool=False):
        self._global_to_local_id = {}
        self._alias_id_to_placeholder = {}
        self._is_legacy_signature = is_legacy_signature

    def alias_global_id(self, global_id: Hashable) -> Hashable:
        if global_id not in self._global_to_local_id:
            self._global_to_local_id[global_id] = len(self._global_to_local_id)
        return self._global_to_local_id[global_id]

    def add_placeholder(self, alias_id: Hashable, variable) -> None:
        self._alias_id_to_placeholder[alias_id] = variable

    def get_placeholder_mapping(self) -> Dict[Hashable, Any]:
        return self._alias_id_to_placeholder

    @property
    def is_legacy_signature(self) -> bool:
        """If the value is from a legacy signature representation.

    Legacy signature representations include tf.function.input_signature and
    ConcreteFunction.structured_input_signature.
    """
        return self._is_legacy_signature