import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MultiRowFormula(CommandBit):
    """A formula with multiple rows."""

    def parserows(self, pos):
        """Parse all rows, finish when no more row ends"""
        self.rows = []
        first = True
        for row in self.iteraterows(pos):
            if first:
                first = False
            else:
                self.addempty()
            row.parsebit(pos)
            self.addrow(row)
        self.size = len(self.rows)

    def iteraterows(self, pos):
        """Iterate over all rows, end when no more row ends"""
        rowseparator = FormulaConfig.array['rowseparator']
        while True:
            pos.pushending(rowseparator, True)
            row = self.factory.create(FormulaRow)
            yield row.setalignments(self.alignments)
            if pos.checkfor(rowseparator):
                self.original += pos.popending(rowseparator)
            else:
                return

    def addempty(self):
        """Add an empty row."""
        row = self.factory.create(FormulaRow).setalignments(self.alignments)
        for index, originalcell in enumerate(self.rows[-1].contents):
            cell = row.createcell(index)
            cell.add(FormulaConstant('\u2005'))
            row.add(cell)
        self.addrow(row)

    def addrow(self, row):
        """Add a row to the contents and to the list of rows."""
        self.rows.append(row)
        self.add(row)