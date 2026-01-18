import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaEquation(CommandBit):
    """A simple numbered equation."""
    piece = 'equation'

    def parsebit(self, pos):
        """Parse the array"""
        self.output = ContentsOutput()
        self.add(self.factory.parsetype(WholeFormula, pos))