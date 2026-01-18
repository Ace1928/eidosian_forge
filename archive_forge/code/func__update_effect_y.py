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
def _update_effect_y(self, *args):
    vp = self._viewport
    if not vp or not self.effect_y:
        return
    if self.effect_y.is_manual:
        sh = vp.height - self._effect_y_start_height
    else:
        sh = vp.height - self.height
    if sh < 1 and (not (self.always_overscroll and self.do_scroll_y)):
        return
    if sh != 0:
        sy = self.effect_y.scroll / sh
        self.scroll_y = -sy
    self._trigger_update_from_scroll()