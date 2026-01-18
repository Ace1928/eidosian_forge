import itertools
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Group, Rect
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColor, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.units import mm
from reportlab.lib.utils import asUnicodeEx, isUnicode
from reportlab.graphics.barcode import qrencoder
def addData(self, value):
    self.qr.addData(value)