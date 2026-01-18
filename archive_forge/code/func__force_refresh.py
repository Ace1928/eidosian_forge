import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def _force_refresh(self, *largs):
    from kivy.base import EventLoop
    win = EventLoop.window
    if win and win.canvas:
        win.canvas.ask_update()