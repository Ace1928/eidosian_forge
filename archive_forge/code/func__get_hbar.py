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
def _get_hbar(self):
    if self._viewport is None:
        return (0, 1.0)
    vw = self._viewport.width
    w = self.width
    if vw < w or vw == 0:
        return (0, 1.0)
    pw = max(0.01, w / float(vw))
    sx = min(1.0, max(0.0, self.scroll_x))
    px = (1.0 - pw) * sx
    return (px, pw)