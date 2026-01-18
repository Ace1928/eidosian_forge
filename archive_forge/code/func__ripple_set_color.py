from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
def _ripple_set_color(self, instance, value):
    if not self.ripple_col_instruction:
        return
    self.ripple_col_instruction.rgba = value