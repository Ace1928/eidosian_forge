import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaCases(MultiRowFormula):
    """A cases statement"""
    piece = 'cases'

    def parsebit(self, pos):
        """Parse the cases"""
        self.output = ContentsOutput()
        self.alignments = ['l', 'l']
        self.parserows(pos)
        for row in self.contents:
            for cell in row.contents:
                cell.output.settag('span class="case align-l"', True)
                cell.contents.append(FormulaConstant('\u2003'))
        array = TaggedBit().complete(self.contents, 'span class="bracketcases"', True)
        brace = BigBracket(len(self.contents), '{', 'l')
        self.contents = brace.getcontents() + [array]