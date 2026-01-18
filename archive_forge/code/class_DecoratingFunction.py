import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class DecoratingFunction(OneParamFunction):
    """A function that decorates some bit of text"""
    commandmap = FormulaConfig.decoratingfunctions

    def parsebit(self, pos):
        """Parse a decorating function"""
        self.type = 'alpha'
        symbol = self.translated
        self.symbol = TaggedBit().constant(symbol, 'span class="symbolover"')
        self.parameter = self.parseparameter(pos)
        self.output = TaggedOutput().settag('span class="withsymbol"')
        self.contents.insert(0, self.symbol)
        self.parameter.output = TaggedOutput().settag('span class="undersymbol"')
        self.simplifyifpossible()