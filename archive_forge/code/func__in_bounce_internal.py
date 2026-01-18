from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def _in_bounce_internal(t, d):
    return 1.0 - AnimationTransition._out_bounce_internal(d - t, d)