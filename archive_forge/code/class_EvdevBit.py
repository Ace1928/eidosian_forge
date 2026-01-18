import os
from functools import total_ordering
from ._clib import Libevdev
import libevdev
@total_ordering
class EvdevBit:
    """
    Base class representing an evdev bit, comprised of a name and a value.
    These two properties are guaranteed to exist on anything describing an
    event code, event type or input property that comes out of libevdev::

        >>> print(libevdev.EV_ABS.name)
        EV_ABS
        >>> print(libevdev.EV_ABS.value)
        3
        >>> print(libevdev.EV_SYN.SYN_REPORT.name)
        SYN_REPORT
        >>> print(libevdev.EV_SYN.SYN_REPORT.value)
        0
        >>> print(libevdev.INPUT_PROP_DIRECT.name)
        INPUT_PROP_DIRECT
        >>> print(libevdev.INPUT_PROP_DIRECT.value)
        1

    .. attribute:: value

        The numeric value of the event code

    .. attribute:: name

        The string name of this event code
    """

    def __repr__(self):
        return '{}:{}'.format(self.name, self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __int__(self):
        return self.value