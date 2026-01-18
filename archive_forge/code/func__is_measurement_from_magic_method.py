from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def _is_measurement_from_magic_method(val: Any) -> Optional[bool]:
    """Uses `is_measurement` magic method to determine if this object is a measurement."""
    getter = getattr(val, '_is_measurement_', None)
    return NotImplemented if getter is None else getter()