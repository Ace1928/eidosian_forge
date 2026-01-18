import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaMatrix(MultiRowFormula):
    """A matrix (array with center alignment)."""
    piece = 'matrix'

    def parsebit(self, pos):
        """Parse the matrix, set alignments to 'c'."""
        self.output = TaggedOutput().settag('span class="array"', False)
        self.valign = 'c'
        self.alignments = ['c']
        self.parserows(pos)