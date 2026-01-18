from .interpolatableHelpers import *
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.recordingPen import (
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.cairoPen import CairoPen
from fontTools.pens.pointPen import (
from fontTools.varLib.interpolatableHelpers import (
from itertools import cycle
from functools import wraps
from io import BytesIO
import cairo
import math
import os
import logging
def _add_listing(self, title, items):
    pad = self.pad
    width = self.width - 2 * self.pad
    height = self.height - 2 * self.pad
    x = y = pad
    self.draw_label(title, x=x, y=y, bold=True, width=width, font_size=self.title_font_size)
    y += self.title_font_size + self.pad
    last_glyphname = None
    for page_no, (glyphname, problems) in items:
        if glyphname == last_glyphname:
            continue
        last_glyphname = glyphname
        if y + self.font_size > height:
            self.show_page()
            y = self.font_size + pad
        self.draw_label(glyphname, x=x + 5 * pad, y=y, width=width - 2 * pad)
        self.draw_label(str(page_no), x=x, y=y, width=4 * pad, align=1)
        y += self.font_size
    self.show_page()