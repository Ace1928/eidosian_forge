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
class InterpolatablePDF(InterpolatablePostscriptLike):

    def __enter__(self):
        self.surface = cairo.PDFSurface(self.out, self.width, self.height)
        self.surface.set_metadata(cairo.PDF_METADATA_CREATOR, 'fonttools varLib.interpolatable')
        self.surface.set_metadata(cairo.PDF_METADATA_CREATE_DATE, '')
        return self