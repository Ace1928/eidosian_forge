import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaRow(FormulaCommand):
    """An array row inside an array"""
    cellseparator = FormulaConfig.array['cellseparator']

    def setalignments(self, alignments):
        self.alignments = alignments
        self.output = TaggedOutput().settag('span class="arrayrow"', True)
        return self

    def parsebit(self, pos):
        """Parse a whole row"""
        index = 0
        pos.pushending(self.cellseparator, optional=True)
        while not pos.finished():
            cell = self.createcell(index)
            cell.parsebit(pos)
            self.add(cell)
            index += 1
            pos.checkskip(self.cellseparator)
        if len(self.contents) == 0:
            self.output = EmptyOutput()

    def createcell(self, index):
        """Create the cell that corresponds to the given index."""
        alignment = self.alignments[index % len(self.alignments)]
        return self.factory.create(FormulaCell).setalignment(alignment)