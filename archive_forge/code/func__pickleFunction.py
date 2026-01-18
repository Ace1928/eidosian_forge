import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def _pickleFunction(f):
    """
    Reduce, in the sense of L{pickle}'s C{object.__reduce__} special method, a
    function object into its constituent parts.

    @param f: The function to reduce.
    @type f: L{types.FunctionType}

    @return: a 2-tuple of a reference to L{_unpickleFunction} and a tuple of
        its arguments, a 1-tuple of the function's fully qualified name.
    @rtype: 2-tuple of C{callable, native string}
    """
    if f.__name__ == '<lambda>':
        raise _UniversalPicklingError(f'Cannot pickle lambda function: {f}')
    return (_unpickleFunction, tuple(['.'.join([f.__module__, f.__qualname__])]))