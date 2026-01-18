from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
def _get_vbar(self):
    if self._viewport is None:
        return (0, 1.0)
    vh = self._viewport.height
    h = self.height
    if vh < h or vh == 0:
        return (0, 1.0)
    ph = max(0.01, h / float(vh))
    sy = min(1.0, max(0.0, self.scroll_y))
    py = (1.0 - ph) * sy
    return (py, ph)