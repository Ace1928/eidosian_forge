import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class SizeText(TaggedText):
    """Sized text"""

    def process(self):
        self.size = self.header[1]
        self.output.tag = 'span class="' + self.size + '"'