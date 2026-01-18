from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
 Returns the full dotted path for a type.

        For example:
        from traits.api import HasTraits
        _get_type_name(HasTraits) == 'traits.has_traits.HasTraits'

        If the type is given as a string (e.g., for lazy loading), it is just
        returned.

        