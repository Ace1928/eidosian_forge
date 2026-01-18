from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def Paragraph(text, style, bulletText=None, frags=None, context=None):
    """ Paragraph(text, style, bulletText=None)
    intended to be like a platypus Paragraph but better.
    """
    if '&' not in text and '<' not in text:
        return FastPara(style, simpletext=text)
    else:
        from reportlab.lib import rparsexml
        parsedpara = rparsexml.parsexmlSimple(text, entityReplacer=None)
        return Para(style, parsedText=parsedpara, bulletText=bulletText, state=None, context=context)