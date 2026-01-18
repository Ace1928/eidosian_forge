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
def _get_do_scroll(self):
    return (self.do_scroll_x, self.do_scroll_y)