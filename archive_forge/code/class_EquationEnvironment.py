import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class EquationEnvironment(MultiRowFormula):
    """A \\begin{}...\\end equation environment with rows and cells."""

    def parsebit(self, pos):
        """Parse the whole environment."""
        self.output = TaggedOutput().settag('span class="environment"', False)
        environment = self.piece.replace('*', '')
        if environment in FormulaConfig.environments:
            self.alignments = FormulaConfig.environments[environment]
        else:
            Trace.error('Unknown equation environment ' + self.piece)
            self.alignments = ['l']
        self.parserows(pos)