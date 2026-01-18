import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def _classOfMethod(methodObject):
    """
    Get the associated class of the given method object.

    @param methodObject: a bound method
    @type methodObject: L{types.MethodType}

    @return: a class
    @rtype: L{type}
    """
    return methodObject.__self__.__class__