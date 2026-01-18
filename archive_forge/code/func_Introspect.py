from typing import Union
from warnings import warn
from .low_level import *
def Introspect(self):
    """Request D-Bus introspection XML for a remote object"""
    return new_method_call(self, 'Introspect')