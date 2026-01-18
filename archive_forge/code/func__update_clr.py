from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def _update_clr(self, dt):
    mode, clr_idx, text = self._upd_clr_list
    try:
        text = min(255.0, max(0.0, float(text)))
        if mode == 'rgb':
            self.color[clr_idx] = text / 255
        else:
            hsv = list(self.hsv[:])
            hsv[clr_idx] = text / 255
            self.color[:3] = hsv_to_rgb(*hsv)
    except ValueError:
        Logger.warning('ColorPicker: invalid value : {}'.format(text))
    finally:
        self._updating_clr = False