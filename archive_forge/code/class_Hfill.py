import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class Hfill(TaggedText):
    """Horizontall fill"""

    def process(self):
        self.output.tag = 'span class="hfill"'