from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _restore_pmap_field_pickle(key_type, value_type, data):
    """Unpickling function for auto-generated PMap field types."""
    type_ = _pmap_field_types[key_type, value_type]
    return _restore_pickle(type_, data)