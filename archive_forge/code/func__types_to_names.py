from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _types_to_names(types):
    """Convert a tuple of types to a human-readable string."""
    return ''.join((get_type(typ).__name__.capitalize() for typ in types))