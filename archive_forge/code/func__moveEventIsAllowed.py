import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def _moveEventIsAllowed(self):
    rateLimit = getConfigOption('mouseRateLimit')
    if rateLimit <= 0:
        return True
    delay = 1000.0 / rateLimit
    if getMillis() - self._lastMoveEventTime >= delay:
        return True
    return False