from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def _reset_canvas(self):
    self.canvas.clear()
    self.arcs = []
    self.sv_idx = 0
    pdv = self._piece_divisions
    ppie = self._pieces_of_pie
    for r in range(pdv):
        for t in range(ppie):
            self.arcs.append(_ColorArc(self._radius * (float(r) / float(pdv)), self._radius * (float(r + 1) / float(pdv)), 2 * pi * (float(t) / float(ppie)), 2 * pi * (float(t + 1) / float(ppie)), origin=self._origin, color=(float(t) / ppie, self.sv_s[self.sv_idx + r][0], self.sv_s[self.sv_idx + r][1], 1)))
            self.canvas.add(self.arcs[-1])