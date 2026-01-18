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
def _get_db_key(self, key, numpoints, orientation_sens):
    n = numpoints and numpoints or self.numpoints
    if n not in self.db:
        self.prepare(n)
    prefix = orientation_sens and 'bound_' or 'inv_'
    return self.db[n][prefix + key]