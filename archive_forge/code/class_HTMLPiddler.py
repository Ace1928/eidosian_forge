import formatter
import string
from types import *
import htmllib
import piddle
class HTMLPiddler:
    """jjk  02/01/00"""

    def __init__(self, html='', start=(0, 0), xLimits=(0, 800), font=None, color=None):
        """instance initializer
            jjk  02/01/00"""
        self.html = html
        self.start = start
        self.xLimits = xLimits
        if not font:
            font = piddle.Font()
        self.font = font
        self.color = color

    def renderOn(self, aPiddleCanvas):
        """draw the text with aPiddleCanvas
            jjk  02/01/00"""
        writer = _HtmlPiddleWriter(self, aPiddleCanvas)
        fmt = formatter.AbstractFormatter(writer)
        parser = _HtmlParser(fmt)
        parser.feed(self.html)
        parser.close()