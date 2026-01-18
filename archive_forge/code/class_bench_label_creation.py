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
class bench_label_creation:
    """Core: label creation (10000 * 10 a-z)"""

    def __init__(self):
        labels = []
        for x in range(10000):
            label = [chr(randint(ord('a'), ord('z'))) for x in range(10)]
            labels.append(''.join(label))
        self.labels = labels

    def run(self):
        o = []
        for x in self.labels:
            o.append(Label(text=x))