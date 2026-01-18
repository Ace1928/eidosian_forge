import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def _methodFunction(classObject, methodName):
    """
    Retrieve the function object implementing a method name given the class
    it's on and a method name.

    @param classObject: A class to retrieve the method's function from.
    @type classObject: L{type}

    @param methodName: The name of the method whose function to retrieve.
    @type methodName: native L{str}

    @return: the function object corresponding to the given method name.
    @rtype: L{types.FunctionType}
    """
    methodObject = getattr(classObject, methodName)
    return methodObject