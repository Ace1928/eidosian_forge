import pyglet
from pyglet.event import EventDispatcher
from pyglet.graphics import Group
from pyglet.text.caret import Caret
from pyglet.text.layout import IncrementalTextLayout
def _update_knob(self, x):
    self._knob_spr.x = max(self._min_knob_x, min(x - self._half_knob_width, self._max_knob_x))
    self._value = abs((self._knob_spr.x - self._min_knob_x) * 100 / (self._min_knob_x - self._max_knob_x))
    self.dispatch_event('on_change', self._value)