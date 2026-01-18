import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class PersistentComponentsDict(dict, PersistentComponents):

    def __init__(self, name):
        dict.__init__(self)
        PersistentComponents.__init__(self, name)