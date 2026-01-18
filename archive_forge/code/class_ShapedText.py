import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ShapedText(TaggedText):
    """Text shaped (italic, slanted)"""

    def process(self):
        self.type = self.header[1]
        if not self.type in TagConfig.shaped:
            Trace.error('Unrecognized shape ' + self.header[1])
            self.output.tag = 'span'
            return
        self.output.tag = TagConfig.shaped[self.type]