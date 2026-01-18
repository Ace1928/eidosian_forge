from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _set_factory(self, factory):
    """ Trait property setter. """
    self._factory = factory