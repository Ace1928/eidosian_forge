from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
def _check_traits(self, cls, interface, error_mode):
    """ Checks that a class implements the traits on an interface.
        """
    missing = set(interface.class_traits()).difference(set(cls.class_traits()))
    if len(missing) > 0:
        return self._handle_error(MISSING_TRAIT % (self._class_name(cls), repr(list(missing))[1:-1], self._class_name(interface)), error_mode)
    return True