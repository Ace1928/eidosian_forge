from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class SimpleStyle:
    """simplified paragraph style without all the fancy stuff"""
    name = 'basic'
    fontName = _baseFontName
    fontSize = 10
    leading = 12
    leftIndent = 0
    rightIndent = 0
    firstLineIndent = 0
    alignment = TA_LEFT
    spaceBefore = 0
    spaceAfter = 0
    bulletFontName = _baseFontName
    bulletFontSize = 10
    bulletIndent = 0
    textColor = black
    backColor = None

    def __init__(self, name, parent=None, **kw):
        mydict = self.__dict__
        if parent:
            for a, b in parent.__dict__.items():
                mydict[a] = b
        for a, b in kw.items():
            mydict[a] = b

    def addAttributes(self, dictionary):
        for key in dictionary.keys():
            value = dictionary[key]
            if value is not None:
                if hasattr(StyleAttributeConverters, key):
                    converter = getattr(StyleAttributeConverters, key)[0]
                    value = converter(value)
                setattr(self, key, value)