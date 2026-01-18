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
class SlideTransition(TransitionBase):
    """Slide Transition, can be used to show a new screen from any direction:
    left, right, up or down.
    """
    direction = OptionProperty('left', options=('left', 'right', 'up', 'down'))
    "Direction of the transition.\n\n    :attr:`direction` is an :class:`~kivy.properties.OptionProperty` and\n    defaults to 'left'. Can be one of 'left', 'right', 'up' or 'down'.\n    "

    def on_progress(self, progression):
        a = self.screen_in
        b = self.screen_out
        manager = self.manager
        x, y = manager.pos
        width, height = manager.size
        direction = self.direction
        al = AnimationTransition.out_quad
        progression = al(progression)
        if direction == 'left':
            a.y = b.y = y
            a.x = x + width * (1 - progression)
            b.x = x - width * progression
        elif direction == 'right':
            a.y = b.y = y
            b.x = x + width * progression
            a.x = x - width * (1 - progression)
        elif direction == 'down':
            a.x = b.x = x
            a.y = y + height * (1 - progression)
            b.y = y - height * progression
        elif direction == 'up':
            a.x = b.x = x
            b.y = y + height * progression
            a.y = y - height * (1 - progression)

    def on_complete(self):
        self.screen_in.pos = self.manager.pos
        self.screen_out.pos = self.manager.pos
        super(SlideTransition, self).on_complete()