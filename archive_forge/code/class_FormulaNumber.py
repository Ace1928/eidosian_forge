import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaNumber(FormulaBit):
    """A string of digits in a formula"""

    def detect(self, pos):
        """Detect a digit"""
        return pos.current().isdigit()

    def parsebit(self, pos):
        """Parse a bunch of digits"""
        digits = pos.glob(lambda: pos.current().isdigit())
        self.add(FormulaConstant(digits))
        self.type = 'number'