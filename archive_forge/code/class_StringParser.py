import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StringParser(Parser):
    """Parses just a string"""

    def parseheader(self, reader):
        """Do nothing, just take note"""
        self.begin = reader.linenumber + 1
        return []

    def parse(self, reader):
        """Parse a single line"""
        contents = reader.currentline()
        reader.nextline()
        return contents