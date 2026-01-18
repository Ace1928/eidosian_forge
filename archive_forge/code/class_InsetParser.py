import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class InsetParser(BoundedParser):
    """Parses a LyX inset"""

    def parse(self, reader):
        """Parse inset parameters into a dictionary"""
        startcommand = ContainerConfig.string['startcommand']
        while reader.currentline() != '' and (not reader.currentline().startswith(startcommand)):
            self.parseparameter(reader)
        return BoundedParser.parse(self, reader)