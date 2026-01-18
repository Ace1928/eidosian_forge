from functools import wraps
from kivy.context import Context
from kivy.base import ExceptionManagerBase
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
def _call_draw(self, dt):
    self.main_clock.schedule_once(self._clock_sandbox_draw, -1)