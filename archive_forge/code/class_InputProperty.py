import os
from functools import total_ordering
from ._clib import Libevdev
import libevdev
class InputProperty(EvdevBit):
    """
    .. warning ::

        Do not instantiate an object of this class, all objects you'll ever need
        are already present in the libevdev namespace. Use :func:`propbit()`
        to get an :class:`InputProperty` from numerical or string values.

    A class representing an evdev input property::

        >>> print(libevdev.INPUT_PROP_DIRECT)
        INPUT_PROP_DIRECT:1
        >> int(libevdev.INPUT_PROP_DIRECT)
        1

    .. attribute:: value

        The numeric value of the property. This value is also returned when
        the object is converted to ``int``.

    .. attribute:: name

        The string name of this property
    """
    __hash__ = super.__hash__

    def __eq__(self, other):
        assert isinstance(other, InputProperty)
        return self.value == other.value