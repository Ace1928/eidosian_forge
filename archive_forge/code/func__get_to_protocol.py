from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _get_to_protocol(self):
    """ Trait property getter. """
    if not self._to_protocol_loaded:
        if isinstance(self._to_protocol, str):
            self._to_protocol = import_symbol(self._to_protocol)
        self._to_protocol_loaded = True
    return self._to_protocol