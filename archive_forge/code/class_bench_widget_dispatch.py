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
class bench_widget_dispatch:
    """Widget: event dispatch (1000 on_update in 10*1000 Widget)"""

    def __init__(self):
        root = Widget()
        for x in range(10):
            parent = Widget()
            for y in range(1000):
                parent.add_widget(Widget())
            root.add_widget(parent)
        self.root = root

    def run(self):
        touch = FakeMotionEvent('fake', 1, [])
        self.root.dispatch('on_touch_down', touch)
        self.root.dispatch('on_touch_move', touch)
        self.root.dispatch('on_touch_up', touch)