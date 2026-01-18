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
class InterpolatablePostscriptLike(InterpolatablePlot):

    def __exit__(self, type, value, traceback):
        self.surface.finish()

    def show_page(self):
        super().show_page()
        self.surface.show_page()