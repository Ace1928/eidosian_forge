import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BoundedParser(ExcludingParser):
    """A parser bound by a final line"""

    def parse(self, reader):
        """Parse everything, including the final line"""
        contents = ExcludingParser.parse(self, reader)
        reader.nextline()
        return contents