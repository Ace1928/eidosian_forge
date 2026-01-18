import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TextFamily(TaggedText):
    """A bit of text from elyxer.a different family"""

    def process(self):
        """Parse the type of family"""
        self.type = self.header[1]
        if not self.type in TagConfig.family:
            Trace.error('Unrecognized family ' + type)
            self.output.tag = 'span'
            return
        self.output.tag = TagConfig.family[self.type]