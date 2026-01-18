import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class AdapterBase:

    def __init__(self, context1, context2):
        self.context1 = context1
        self.context2 = context2