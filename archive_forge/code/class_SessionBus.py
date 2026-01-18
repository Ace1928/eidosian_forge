from __future__ import generators
from dbus.exceptions import DBusException
from _dbus_bindings import (
from dbus.bus import BusConnection
from dbus.lowlevel import SignalMessage
from dbus._compat import is_py2
class SessionBus(Bus):
    """The session (current login) message bus."""

    def __new__(cls, private=False, mainloop=None):
        """Return a connection to the session bus.

        :Parameters:
            `private` : bool
                If true, never return an existing shared instance, but instead
                return a private connection.
            `mainloop` : dbus.mainloop.NativeMainLoop
                The main loop to use. The default is to use the default
                main loop if one has been set up, or raise an exception
                if none has been.
        """
        return Bus.__new__(cls, Bus.TYPE_SESSION, private=private, mainloop=mainloop)