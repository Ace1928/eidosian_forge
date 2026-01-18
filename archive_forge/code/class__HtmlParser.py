import formatter
import string
from types import *
import htmllib
import piddle
class _HtmlParser(htmllib.HTMLParser):

    def anchor_bgn(self, href, name, type):
        htmllib.HTMLParser.anchor_bgn(self, href, name, type)
        self.formatter.writer.anchor_bgn(href, name, type)

    def anchor_end(self):
        htmllib.HTMLParser.anchor_end(self)
        self.formatter.writer.anchor_end()