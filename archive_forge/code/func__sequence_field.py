from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def _sequence_field(checked_class, item_type, optional, initial, invariant=PFIELD_NO_INVARIANT, item_invariant=PFIELD_NO_INVARIANT):
    """
    Create checked field for either ``PSet`` or ``PVector``.

    :param checked_class: ``CheckedPSet`` or ``CheckedPVector``.
    :param item_type: The required type for the items in the set.
    :param optional: If true, ``None`` can be used as a value for
        this field.
    :param initial: Initial value to pass to factory.

    :return: A ``field`` containing a checked class.
    """
    TheType = _make_seq_field_type(checked_class, item_type, item_invariant)
    if optional:

        def factory(argument, _factory_fields=None, ignore_extra=False):
            if argument is None:
                return None
            else:
                return TheType.create(argument, _factory_fields=_factory_fields, ignore_extra=ignore_extra)
    else:
        factory = TheType.create
    return field(type=optional_type(TheType) if optional else TheType, factory=factory, mandatory=True, invariant=invariant, initial=factory(initial))