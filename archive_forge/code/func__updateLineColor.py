import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def _updateLineColor(self, color):
    """Triggered when someone assigns to defaultLineColor"""
    self.pdf.setStrokeColorRGB(color.red, color.green, color.blue)