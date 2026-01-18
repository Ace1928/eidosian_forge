import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BoundedDummy(Parser):
    """A bound parser that ignores everything"""

    def parse(self, reader):
        """Parse the contents of the container"""
        self.parseending(reader, lambda: reader.nextline())
        reader.nextline()
        return []