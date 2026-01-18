from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.animation import Animation
from kivy.uix.stencilview import StencilView
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import BooleanProperty, OptionProperty, AliasProperty, \
def _prev_slide(self):
    slides = self.slides
    len_slides = len(slides)
    index = self.index
    if len_slides < 2:
        return None
    if self.loop and index == 0:
        return slides[-1]
    if index > 0:
        return slides[index - 1]