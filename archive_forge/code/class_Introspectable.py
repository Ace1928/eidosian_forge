from typing import Union
from warnings import warn
from .low_level import *
class Introspectable(MessageGenerator):
    interface = 'org.freedesktop.DBus.Introspectable'

    def Introspect(self):
        """Request D-Bus introspection XML for a remote object"""
        return new_method_call(self, 'Introspect')