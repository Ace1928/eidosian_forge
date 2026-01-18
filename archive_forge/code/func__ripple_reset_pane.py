from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
def _ripple_reset_pane(self):
    self.ripple_rad = self.ripple_rad_default
    self.ripple_pane.clear()