from typing import Union
from warnings import warn
from .low_level import *
class MessageGenerator:
    """Subclass this to define the methods available on a DBus interface.
    
    jeepney.bindgen can automatically create subclasses using introspection.
    """

    def __init__(self, object_path, bus_name):
        self.object_path = object_path
        self.bus_name = bus_name

    def __repr__(self):
        return '{}({!r}, bus_name={!r})'.format(type(self).__name__, self.object_path, self.bus_name)