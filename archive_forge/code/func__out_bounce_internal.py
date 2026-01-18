from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def _out_bounce_internal(t, d):
    p = t / d
    if p < 1.0 / 2.75:
        return 7.5625 * p * p
    elif p < 2.0 / 2.75:
        p -= 1.5 / 2.75
        return 7.5625 * p * p + 0.75
    elif p < 2.5 / 2.75:
        p -= 2.25 / 2.75
        return 7.5625 * p * p + 0.9375
    else:
        p -= 2.625 / 2.75
        return 7.5625 * p * p + 0.984375