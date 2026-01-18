from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
@property
def _prev_equals_next(self):
    return self.loop and len(self.slides) == 2