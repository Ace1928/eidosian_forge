import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class NewPage(Newline):
    """A new page"""

    def process(self):
        """Process contents"""
        self.html = ['<p><br/>\n</p>\n']