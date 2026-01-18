import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class InsetLength(BlackBox):
    """A length measure inside an inset."""

    def process(self):
        self.length = self.header[1]