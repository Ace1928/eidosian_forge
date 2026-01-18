from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
def _remove_out_canvas(self, *args):
    if self.screen_out and self.screen_out.canvas in self.manager.canvas.children and (self.screen_out not in self.manager.children):
        self.manager.canvas.remove(self.screen_out.canvas)