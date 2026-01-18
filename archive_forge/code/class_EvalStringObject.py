from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class EvalStringObject:
    """this will only work if rml2pdf is present"""
    tagname = 'evalString'

    def __init__(self, attdict, content, extra, context):
        if not attdict:
            attdict = {}
        self.attdict = attdict
        self.content = content
        self.context = context
        self.extra = extra

    def getOp(self, tuple, engine):
        from rlextra.rml2pdf.rml2pdf import Controller
        op = self.op = Controller.processTuple(tuple, self.context, {})
        return op

    def width(self, engine):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        content = self.content
        if not content:
            content = []
        tuple = (self.tagname, self.attdict, content, self.extra)
        op = self.op = self.getOp(tuple, engine)
        s = str(op)
        return stringWidth(s, engine.fontName, engine.fontSize)

    def execute(self, engine, textobject, canvas):
        textobject.textOut(str(self.op))