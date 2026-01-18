import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class AbsoluteAxis(Control):
    """An axis whose value represents a physical measurement from the device.

    The value is advertised to range over ``minimum`` and ``maximum``.

    :Ivariables:
        `minimum` : float
            Minimum advertised value.
        `maximum` : float
            Maximum advertised value.
    """
    X = 'x'
    Y = 'y'
    Z = 'z'
    RX = 'rx'
    RY = 'ry'
    RZ = 'rz'
    HAT = 'hat'
    HAT_X = 'hat_x'
    HAT_Y = 'hat_y'

    def __init__(self, name, minimum, maximum, raw_name=None, inverted=False):
        super().__init__(name, raw_name, inverted)
        self.min = minimum
        self.max = maximum