import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MacroParser(FormulaParser):
    """A parser for a formula macro."""

    def parseheader(self, reader):
        """See if the formula is inlined"""
        self.begin = reader.linenumber + 1
        return ['inline']

    def parse(self, reader):
        """Parse the formula until the end"""
        formula = self.parsemultiliner(reader, self.parent.start, self.ending)
        reader.nextline()
        return formula