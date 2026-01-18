from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class ChannelMixEffect(EffectBase):
    """Mixes the color channels of the input according to the order
    property. Channels may be arbitrarily rearranged or repeated."""
    order = ListProperty([1, 2, 0])
    'The new sorted order of the rgb channels.\n\n    order is a :class:`~kivy.properties.ListProperty` and defaults to\n    [1, 2, 0], corresponding to (g, b, r).\n    '

    def __init__(self, *args, **kwargs):
        super(ChannelMixEffect, self).__init__(*args, **kwargs)
        self.do_glsl()

    def on_order(self, *args):
        self.do_glsl()

    def do_glsl(self):
        letters = [{0: 'x', 1: 'y', 2: 'z'}[i] for i in self.order]
        self.glsl = effect_mix.format(*letters)