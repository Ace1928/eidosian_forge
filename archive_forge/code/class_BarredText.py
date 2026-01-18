import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BarredText(TaggedText):
    """Text with a bar somewhere"""

    def process(self):
        """Parse the type of bar"""
        self.type = self.header[1]
        if not self.type in TagConfig.barred:
            Trace.error('Unknown bar type ' + self.type)
            self.output.tag = 'span'
            return
        self.output.tag = TagConfig.barred[self.type]