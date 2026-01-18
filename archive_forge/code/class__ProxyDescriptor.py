from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
class _ProxyDescriptor:
    """
    A descriptor which will proxy attribute access, mutation, and
    deletion to the L{_ProxyDescriptor.originalAttribute} of the
    object it is being accessed from.

    @ivar attributeName: the name of the attribute which this descriptor will
        retrieve from instances' C{original} attribute.
    @type attributeName: C{str}

    @ivar originalAttribute: name of the attribute of the proxy where the
        original object is stored.
    @type originalAttribute: C{str}
    """

    def __init__(self, attributeName, originalAttribute):
        self.attributeName = attributeName
        self.originalAttribute = originalAttribute

    def __get__(self, oself, type=None):
        """
        Retrieve the C{self.attributeName} property from I{oself}.
        """
        if oself is None:
            return _ProxiedClassMethod(self.attributeName, self.originalAttribute)
        original = getattr(oself, self.originalAttribute)
        return getattr(original, self.attributeName)

    def __set__(self, oself, value):
        """
        Set the C{self.attributeName} property of I{oself}.
        """
        original = getattr(oself, self.originalAttribute)
        setattr(original, self.attributeName, value)

    def __delete__(self, oself):
        """
        Delete the C{self.attributeName} property of I{oself}.
        """
        original = getattr(oself, self.originalAttribute)
        delattr(original, self.attributeName)