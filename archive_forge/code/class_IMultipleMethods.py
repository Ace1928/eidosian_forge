from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class IMultipleMethods(Interface):
    """
    An interface with multiple methods.
    """

    def methodOne():
        """
        The first method. Should return 1.
        """

    def methodTwo():
        """
        The second method. Should return 2.
        """