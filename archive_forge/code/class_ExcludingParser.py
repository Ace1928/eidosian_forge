import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ExcludingParser(Parser):
    """A parser that excludes the final line"""

    def parse(self, reader):
        """Parse everything up to (and excluding) the final line"""
        contents = []
        self.parseending(reader, lambda: self.parsecontainer(reader, contents))
        return contents