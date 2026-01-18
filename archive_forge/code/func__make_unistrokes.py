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
def _make_unistrokes(self):
    unistrokes = []
    unistrokes_append = unistrokes.append
    self_strokes = self.strokes
    for r in self._orders:
        b = 0
        while b < pow(2, len(r)):
            unistroke = []
            unistroke_append = unistroke.append
            for i in xrange(0, len(r)):
                pts = self_strokes[r[i]][:]
                if b >> i & 1 == 1:
                    pts.reverse()
                unistroke_append(None)
                unistroke[-1:] = pts
            unistrokes_append(unistroke)
            b += 1
    return unistrokes