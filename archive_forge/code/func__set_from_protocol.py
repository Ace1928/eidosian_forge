from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _set_from_protocol(self, from_protocol):
    """ Trait property setter. """
    self._from_protocol = from_protocol