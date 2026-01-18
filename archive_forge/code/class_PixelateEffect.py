from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class PixelateEffect(EffectBase):
    """Pixelates the input according to its
    :attr:`~PixelateEffect.pixel_size`"""
    pixel_size = NumericProperty(10)
    "\n    Sets the size of a new 'pixel' in the effect, in terms of number of\n    'real' pixels.\n\n    pixel_size is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 10.\n    "

    def __init__(self, *args, **kwargs):
        super(PixelateEffect, self).__init__(*args, **kwargs)
        self.do_glsl()

    def on_pixel_size(self, *args):
        self.do_glsl()

    def do_glsl(self):
        self.glsl = effect_pixelate.format(float(self.pixel_size))