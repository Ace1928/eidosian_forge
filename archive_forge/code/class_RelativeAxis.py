import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class RelativeAxis(Control):
    """An axis whose value represents a relative change from the previous
    value.
    """
    X = 'x'
    Y = 'y'
    Z = 'z'
    RX = 'rx'
    RY = 'ry'
    RZ = 'rz'
    WHEEL = 'wheel'

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.dispatch_event('on_change', value)