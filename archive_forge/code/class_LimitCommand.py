import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LimitCommand(EmptyCommand):
    """A command which accepts limits above and below, in display mode."""
    commandmap = FormulaConfig.limitcommands

    def parsebit(self, pos):
        """Parse a limit command."""
        pieces = BigSymbol(self.translated).getpieces()
        self.output = TaggedOutput().settag('span class="limits"')
        for piece in pieces:
            self.contents.append(TaggedBit().constant(piece, 'span class="limit"'))