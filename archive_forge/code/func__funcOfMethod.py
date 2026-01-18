import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def _funcOfMethod(methodObject):
    """
    Get the associated function of the given method object.

    @param methodObject: a bound method
    @type methodObject: L{types.MethodType}

    @return: the function implementing C{methodObject}
    @rtype: L{types.FunctionType}
    """
    return methodObject.__func__