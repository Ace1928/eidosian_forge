import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class DeviceOpenException(DeviceException):
    pass