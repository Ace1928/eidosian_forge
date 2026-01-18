import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
class BusName(object):
    """A base class for exporting your own Named Services across the Bus.

    When instantiated, objects of this class attempt to claim the given
    well-known name on the given bus for the current process. The name is
    released when the BusName object becomes unreferenced.

    If a well-known name is requested multiple times, multiple references
    to the same BusName object will be returned.

    :Caveats:

        - Assumes that named services are only ever requested using this class -
          if you request names from the bus directly, confusion may occur.
        - Does not handle queueing.
    """

    def __new__(cls, name, bus=None, allow_replacement=False, replace_existing=False, do_not_queue=False):
        """Constructor, which may either return an existing cached object
        or a new object.

        :Parameters:
            `name` : str
                The well-known name to be advertised
            `bus` : dbus.Bus
                A Bus on which this service will be advertised.

                Omitting this parameter or setting it to None has been
                deprecated since version 0.82.1. For backwards compatibility,
                if this is done, the global shared connection to the session
                bus will be used.

            `allow_replacement` : bool
                If True, other processes trying to claim the same well-known
                name will take precedence over this one.
            `replace_existing` : bool
                If True, this process can take over the well-known name
                from other processes already holding it.
            `do_not_queue` : bool
                If True, this service will not be placed in the queue of
                services waiting for the requested name if another service
                already holds it.
        """
        validate_bus_name(name, allow_well_known=True, allow_unique=False)
        if bus is None:
            import warnings
            warnings.warn('Omitting the "bus" parameter to dbus.service.BusName.__init__ is deprecated', DeprecationWarning, stacklevel=2)
            bus = SessionBus()
        if name in bus._bus_names:
            return bus._bus_names[name]
        name_flags = (allow_replacement and _dbus_bindings.NAME_FLAG_ALLOW_REPLACEMENT or 0) | (replace_existing and _dbus_bindings.NAME_FLAG_REPLACE_EXISTING or 0) | (do_not_queue and _dbus_bindings.NAME_FLAG_DO_NOT_QUEUE or 0)
        retval = bus.request_name(name, name_flags)
        if retval == _dbus_bindings.REQUEST_NAME_REPLY_PRIMARY_OWNER:
            pass
        elif retval == _dbus_bindings.REQUEST_NAME_REPLY_IN_QUEUE:
            pass
        elif retval == _dbus_bindings.REQUEST_NAME_REPLY_EXISTS:
            raise NameExistsException(name)
        elif retval == _dbus_bindings.REQUEST_NAME_REPLY_ALREADY_OWNER:
            pass
        else:
            raise RuntimeError('requesting bus name %s returned unexpected value %s' % (name, retval))
        bus_name = object.__new__(cls)
        bus_name._bus = bus
        bus_name._name = name
        bus._bus_names[name] = bus_name
        return bus_name

    def __init__(self, *args, **keywords):
        pass

    def __del__(self):
        self._bus.release_name(self._name)
        pass

    def get_bus(self):
        """Get the Bus this Service is on"""
        return self._bus

    def get_name(self):
        """Get the name of this service"""
        return self._name

    def __repr__(self):
        return '<dbus.service.BusName %s on %r at %#x>' % (self._name, self._bus, id(self))
    __str__ = __repr__