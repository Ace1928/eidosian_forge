import sys, time
from reportlab import Version as __RL_Version__
from reportlab.graphics.barcode.common import *
from reportlab.graphics.barcode.code39 import *
from reportlab.graphics.barcode.code93 import *
from reportlab.graphics.barcode.code128 import *
from reportlab.graphics.barcode.usps import *
from reportlab.graphics.barcode.usps4s import USPS_4State
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.barcode.dmtx import DataMatrixWidget, pylibdmtx
from reportlab.platypus import Spacer, SimpleDocTemplate, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.flowables import XBox, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Line
from reportlab.graphics.barcode import getCodeNames, createBarcodeDrawing, createBarcodeImageInMemory
def addCross(d, x, y, w=5, h=5, strokeColor='black', strokeWidth=0.5):
    w *= 0.5
    h *= 0.5
    d.add(Line(x - w, y, x + w, y, strokeWidth=0.5, strokeColor=colors.blue))
    d.add(Line(x, y - h, x, y + h, strokeWidth=0.5, strokeColor=colors.blue))