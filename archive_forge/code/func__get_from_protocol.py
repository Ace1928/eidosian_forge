from traits.api import Any, Bool, HasTraits, Property
from traits.util.api import import_symbol
def _get_from_protocol(self):
    """ Trait property getter. """
    if not self._from_protocol_loaded:
        if isinstance(self._from_protocol, str):
            self._from_protocol = import_symbol(self._from_protocol)
        self._from_protocol_loaded = True
    return self._from_protocol