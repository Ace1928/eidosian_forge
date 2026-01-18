from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
The :attr:`wheel` holds the color wheel.

    :attr:`wheel` is an :class:`~kivy.properties.ObjectProperty` and
    defaults to None.
    