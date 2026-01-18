import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def _draw_line(self, value, line_num, texture, viewport_pos, line_height, miny, maxy, x, y, base_dir, halign, rects, auto_halign_r):
    size = list(texture.size)
    texcoords = texture.tex_coords[:]
    padding_left, padding_top, padding_right, padding_bottom = self.padding
    viewport_width = self.width - padding_left - padding_right
    viewport_height = self.height - padding_top - padding_bottom
    texture_width, texture_height = size
    original_height, original_width = tch, tcw = texcoords[1:3]
    if viewport_pos:
        tcx, tcy = viewport_pos
        tcx = tcx / texture_width * original_width
        tcy = tcy / texture_height * original_height
    else:
        tcx, tcy = (0, 0)
    if texture_width * (1 - tcx) < viewport_width:
        tcw = tcw - tcx
        texture_width = tcw * texture_width
    elif viewport_width < texture_width:
        tcw = viewport_width / texture_width * tcw
        texture_width = viewport_width
    if viewport_height < texture_height:
        tch = viewport_height / texture_height * tch
        texture_height = viewport_height
    if y > maxy:
        viewport_height = maxy - y + line_height
        tch = viewport_height / line_height * original_height
        tcy = original_height - tch
        texture_height = viewport_height
    if y - line_height < miny:
        diff = miny - (y - line_height)
        y += diff
        viewport_height = line_height - diff
        tch = viewport_height / line_height * original_height
        texture_height = viewport_height
    if tcw < 0:
        return y
    top_left_corner = (tcx, tcy + tch)
    top_right_corner = (tcx + tcw, tcy + tch)
    bottom_right_corner = (tcx + tcw, tcy)
    bottom_left_corner = (tcx, tcy)
    texcoords = top_left_corner + top_right_corner + bottom_right_corner + bottom_left_corner
    xoffset = 0
    if not base_dir:
        base_dir = self._resolved_base_dir = Label.find_base_direction(value)
        if base_dir and halign == 'auto':
            auto_halign_r = 'rtl' in base_dir
    if halign == 'center':
        xoffset = int((viewport_width - texture_width) / 2.0)
    elif halign == 'right' or auto_halign_r:
        xoffset = max(0, int(viewport_width - texture_width))
    rect = rects[line_num]
    rect.pos = (int(xoffset + x), int(y - line_height))
    rect.size = (texture_width, texture_height)
    rect.texture = texture
    rect.tex_coords = texcoords
    self.canvas.add(rect)
    return y