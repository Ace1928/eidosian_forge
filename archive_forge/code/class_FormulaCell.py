import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaCell(FormulaCommand):
    """An array cell inside a row"""

    def setalignment(self, alignment):
        self.alignment = alignment
        self.output = TaggedOutput().settag('span class="arraycell align-' + alignment + '"', True)
        return self

    def parsebit(self, pos):
        self.factory.clearskipped(pos)
        if pos.finished():
            return
        self.add(self.factory.parsetype(WholeFormula, pos))