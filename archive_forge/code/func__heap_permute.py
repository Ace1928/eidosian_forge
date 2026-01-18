import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def _heap_permute(self, n):
    self_order = self._order
    if n == 1:
        self._orders.append(self_order[:])
    else:
        i = 0
        for i in xrange(0, n):
            self._heap_permute(n - 1)
            if n % 2 == 1:
                tmp = self_order[0]
                self_order[0] = self_order[n - 1]
                self_order[n - 1] = tmp
            else:
                tmp = self_order[i]
                self_order[i] = self_order[n - 1]
                self_order[n - 1] = tmp