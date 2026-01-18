import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class DocPara(DocAssign):
    """at wrap time create a paragraph with the value of expr as text
    if format is specified it should use %(__expr__)s for string interpolation
    of the expression expr (if any). It may also use %(name)s interpolations
    for other variables in the namespace.
    suitable defaults will be used if style and klass are None
    """

    def __init__(self, expr, format=None, style=None, klass=None, escape=True):
        Flowable.__init__(self)
        self.expr = expr
        self.format = format
        self.style = style
        self.klass = klass
        self.escape = escape

    def func(self):
        expr = self.expr
        if expr:
            if not isinstance(expr, str):
                expr = str(expr)
            return self._doctemplateAttr('docEval')(expr)

    def add_content(self, *args):
        self._doctemplateAttr('frame').add_generated_content(*args)

    def get_value(self, aW, aH):
        value = self.funcWrap(aW, aH)
        if self.format:
            NS = self._doctemplateAttr('_nameSpace').copy()
            NS.update(dict(availableWidth=aW, availableHeight=aH))
            NS['__expr__'] = value
            value = self.format % NS
        else:
            value = str(value)
        return value

    def wrap(self, aW, aH):
        value = self.get_value(aW, aH)
        P = self.klass
        if not P:
            from reportlab.platypus.paragraph import Paragraph as P
        style = self.style
        if not style:
            from reportlab.lib.styles import getSampleStyleSheet
            style = getSampleStyleSheet()['Code']
        if self.escape:
            from xml.sax.saxutils import escape
            value = escape(value)
        self.add_content(P(value, style=style))
        return (0, 0)