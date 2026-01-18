from typing import Union
from warnings import warn
from .low_level import *
class DBusAddress:
    """This identifies the object and interface a message is for.

    e.g. messages to display desktop notifications would have this address::

        DBusAddress('/org/freedesktop/Notifications',
                    bus_name='org.freedesktop.Notifications',
                    interface='org.freedesktop.Notifications')
    """

    def __init__(self, object_path, bus_name=None, interface=None):
        self.object_path = object_path
        self.bus_name = bus_name
        self.interface = interface

    def __repr__(self):
        return '{}({!r}, bus_name={!r}, interface={!r})'.format(type(self).__name__, self.object_path, self.bus_name, self.interface)

    def with_interface(self, interface):
        return type(self)(self.object_path, self.bus_name, interface)