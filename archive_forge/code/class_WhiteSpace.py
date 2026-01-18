import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class WhiteSpace(FormulaBit):
    """Some white space inside a formula."""

    def detect(self, pos):
        """Detect the white space."""
        return pos.current().isspace()

    def parsebit(self, pos):
        """Parse all whitespace."""
        self.original += pos.skipspace()

    def __unicode__(self):
        """Return a printable representation."""
        return 'Whitespace: *' + self.original + '*'