import os
from math import cos, pi, sin
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
from . import pdfgen, pdfgeom, pdfmetrics
def _updateLineWidth(self, width):
    """Triggered when someone assigns to defaultLineWidth"""
    self.pdf.setLineWidth(width)