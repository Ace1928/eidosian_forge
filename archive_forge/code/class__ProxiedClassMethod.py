from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
class _ProxiedClassMethod:
    """
    A proxied class method.

    @ivar methodName: the name of the method which this should invoke when
        called.
    @type methodName: L{str}

    @ivar __name__: The name of the method being proxied (the same as
        C{methodName}).
    @type __name__: L{str}

    @ivar originalAttribute: name of the attribute of the proxy where the
        original object is stored.
    @type originalAttribute: L{str}
    """

    def __init__(self, methodName, originalAttribute):
        self.methodName = self.__name__ = methodName
        self.originalAttribute = originalAttribute

    def __call__(self, oself, *args, **kw):
        """
        Invoke the specified L{methodName} method of the C{original} attribute
        for proxyForInterface.

        @param oself: an instance of a L{proxyForInterface} object.

        @return: the result of the underlying method.
        """
        original = getattr(oself, self.originalAttribute)
        actualMethod = getattr(original, self.methodName)
        return actualMethod(*args, **kw)