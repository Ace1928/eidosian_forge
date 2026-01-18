from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet

    Converts the object and all the objects nested in it in its
    immutable counterparts.
    
    The conversion map is in getFreezeConversionMap().
    
    You can register a new conversion using `register()` You can also
    pass a map of custom converters with `custom_converters` and a map
    of custom inverse converters with `custom_inverse_converters`,
    without using `register()`.
    
    By default, if the type is not registered and has a `__dict__`
    attribute, it's converted to the `frozendict` of that `__dict__`.
    
    This function assumes that hashable == immutable (that is not
    always true).
    
    This function uses recursion, with all the limits of recursions in
    Python.
    
    Where is a good old tail call when you need it?
    