import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class _Component:

    def __repr__(self):
        return 'TEST'