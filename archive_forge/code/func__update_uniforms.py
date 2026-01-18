from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
def _update_uniforms(self, *args):
    if self.fbo is None:
        return
    for key, value in self.uniforms.items():
        self.fbo[key] = value