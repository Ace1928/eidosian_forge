import os, sys
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import asNative, base64_decodebytes
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import *
import unittest
from reportlab.rl_config import register_reset
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
class ShapesTestCase(unittest.TestCase):
    """Test generating all kinds of shapes."""

    def setUp(self):
        """Prepare some things before the tests start."""
        self.funcNames = getAllFunctionDrawingNames()
        self.drawings = []

    def tearDown(self):
        """Do what has to be done after the tests are over."""
        writePDF(self.drawings)

    def testAllDrawings(self):
        """Make a list of drawings."""
        for f in self.funcNames:
            if f[0:10] == 'getDrawing':
                _evalFuncDrawing(f, self.drawings)