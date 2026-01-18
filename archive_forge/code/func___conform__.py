from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def __conform__(self, interface):
    return self.getComponent(interface)