import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MacroParameter(FormulaBit):
    """A parameter from elyxer.a macro."""

    def detect(self, pos):
        """Find a macro parameter: #n."""
        return pos.checkfor('#')

    def parsebit(self, pos):
        """Parse the parameter: #n."""
        if not pos.checkskip('#'):
            Trace.error('Missing parameter start #.')
            return
        self.number = int(pos.skipcurrent())
        self.original = '#' + str(self.number)
        self.contents = [TaggedBit().constant('#' + str(self.number), 'span class="unknown"')]