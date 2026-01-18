import re
import traitlets
import datetime as dt
class TypedTuple(traitlets.Container):
    """A trait for a tuple of any length with type-checked elements."""
    klass = tuple
    _cast_types = (list,)