import time
import os
import libevdev
from ._clib import Libevdev, UinputDevice
from ._clib import READ_FLAG_SYNC, READ_FLAG_NORMAL, READ_FLAG_FORCE_SYNC, READ_FLAG_BLOCKING
from .event import InputEvent
from .const import InputProperty
class EventsDroppedException(Exception):
    """
    Notification that the device has dropped events, raised in response to a
    EV_SYN SYN_DROPPED event.

    This exception is raised AFTER the EV_SYN, SYN_DROPPED event has been
    passed on. If SYN_DROPPED events are processed manually, then this
    exception can be ignored.

    Once received (or in response to a SYN_DROPPED event) a caller should
    call device.sync() and process the events accordingly (if any).

    Example::

            fd = open("/dev/input/event0", "rb")
            ctx = libevdev.Device(fd)

            while True:
                try:
                    for e in ctx.events():
                        print(e):
                except EventsDroppedException:
                    print('State lost, re-synching:')
                    for e in ctx.sync():
                        print(e)
    """
    pass