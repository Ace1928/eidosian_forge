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
def _add_result(self, gesture, dist, tpl, res):
    if tpl <= len(res):
        n = gesture.templates[tpl].name
    else:
        return 0.0
    if n not in self.results or dist < self.results[n]['dist']:
        self.results[n] = {'name': n, 'dist': dist, 'gesture': gesture, 'best_template': tpl, 'template_results': res}
        if not dist:
            self.results[n]['score'] = 1.0
        else:
            self.results[n]['score'] = 1.0 - dist / pi
        self.dispatch('on_result', self.results[n])
        return self.results[n]['score']
    else:
        return 0.0