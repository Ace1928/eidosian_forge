from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
def _get_public_methods(self, cls):
    """ Returns all public methods on a class.

            Returns a dictionary containing all public methods keyed by name.
        """
    public_methods = {}
    for c in getmro(cls):
        if c is HasTraits:
            break
        for name, value in c.__dict__.items():
            if not name.startswith('_') and type(value) is FunctionType:
                if name not in public_methods:
                    public_methods[name] = value
    return public_methods