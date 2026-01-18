import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BigBracket(BigSymbol):
    """A big bracket generator."""

    def __init__(self, size, bracket, alignment='l'):
        """Set the size and symbol for the bracket."""
        self.size = size
        self.original = bracket
        self.alignment = alignment
        self.pieces = None
        if bracket in FormulaConfig.bigbrackets:
            self.pieces = FormulaConfig.bigbrackets[bracket]

    def getpiece(self, index):
        """Return the nth piece for the bracket."""
        function = getattr(self, 'getpiece' + str(len(self.pieces)))
        return function(index)

    def getpiece1(self, index):
        """Return the only piece for a single-piece bracket."""
        return self.pieces[0]

    def getpiece3(self, index):
        """Get the nth piece for a 3-piece bracket: parenthesis or square bracket."""
        if index == 0:
            return self.pieces[0]
        if index == self.size - 1:
            return self.pieces[-1]
        return self.pieces[1]

    def getpiece4(self, index):
        """Get the nth piece for a 4-piece bracket: curly bracket."""
        if index == 0:
            return self.pieces[0]
        if index == self.size - 1:
            return self.pieces[3]
        if index == (self.size - 1) / 2:
            return self.pieces[2]
        return self.pieces[1]

    def getcell(self, index):
        """Get the bracket piece as an array cell."""
        piece = self.getpiece(index)
        span = 'span class="bracket align-' + self.alignment + '"'
        return TaggedBit().constant(piece, span)

    def getcontents(self):
        """Get the bracket as an array or as a single bracket."""
        if self.size == 1 or not self.pieces:
            return self.getsinglebracket()
        rows = []
        for index in range(self.size):
            cell = self.getcell(index)
            rows.append(TaggedBit().complete([cell], 'span class="arrayrow"'))
        return [TaggedBit().complete(rows, 'span class="array"')]

    def getsinglebracket(self):
        """Return the bracket as a single sign."""
        if self.original == '.':
            return [TaggedBit().constant('', 'span class="emptydot"')]
        return [TaggedBit().constant(self.original, 'span class="symbol"')]