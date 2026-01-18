import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FilteredOutput(ContentsOutput):
    """Returns the output in the contents, but filtered:"""
    'some strings are replaced by others.'

    def __init__(self):
        """Initialize the filters."""
        self.filters = []

    def addfilter(self, original, replacement):
        """Add a new filter: replace the original by the replacement."""
        self.filters.append((original, replacement))

    def gethtml(self, container):
        """Return the HTML code"""
        result = []
        html = ContentsOutput.gethtml(self, container)
        for line in html:
            result.append(self.filter(line))
        return result

    def filter(self, line):
        """Filter a single line with all available filters."""
        for original, replacement in self.filters:
            if original in line:
                line = line.replace(original, replacement)
        return line