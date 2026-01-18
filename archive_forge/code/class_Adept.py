from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@implementer(IAdept)
class Adept(components.Adapter):

    def __init__(self, orig):
        self.original = orig
        self.num = 0

    def adaptorFunc(self):
        self.num = self.num + 1
        return (self.num, self.original.inc())