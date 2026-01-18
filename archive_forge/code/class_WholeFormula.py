import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class WholeFormula(FormulaBit):
    """Parse a whole formula"""

    def detect(self, pos):
        """Not outside the formula is enough."""
        return not pos.finished()

    def parsebit(self, pos):
        """Parse with any formula bit"""
        while not pos.finished():
            self.add(self.factory.parseany(pos))