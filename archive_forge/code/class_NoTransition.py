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
class NoTransition(TransitionBase):
    """No transition, instantly switches to the next screen with no delay or
    animation.

    .. versionadded:: 1.8.0
    """
    duration = NumericProperty(0.0)

    def on_complete(self):
        self.screen_in.pos = self.manager.pos
        self.screen_out.pos = self.manager.pos
        super(NoTransition, self).on_complete()