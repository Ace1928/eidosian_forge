from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class ScanlinesEffect(EffectBase):
    """Adds scanlines to the input."""

    def __init__(self, *args, **kwargs):
        super(ScanlinesEffect, self).__init__(*args, **kwargs)
        self.glsl = effect_postprocessing