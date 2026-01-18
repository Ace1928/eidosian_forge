import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StrikeOut(TaggedText):
    """Striken out text."""

    def process(self):
        """Set the output tag to strike."""
        self.output.tag = 'strike'