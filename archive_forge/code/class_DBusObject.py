from typing import Union
from warnings import warn
from .low_level import *
class DBusObject(DBusAddress):

    def __init__(self, object_path, bus_name=None, interface=None):
        super().__init__(object_path, bus_name, interface)
        warn('Deprecated alias, use DBusAddress instead', stacklevel=2)