from __future__ import print_function
import os
import sys
import json
import kivy
import gc
from time import clock, time, ctime
from random import randint
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import RenderContext
from kivy.input.motionevent import MotionEvent
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.compat import PY2
class bench_widget_creation_with_root:
    """Widget: creation (10000 Widget + 1 root)"""

    def run(self):
        o = Widget()
        for x in range(10000):
            o.add_widget(Widget())