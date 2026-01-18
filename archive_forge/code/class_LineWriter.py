import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LineWriter(object):
    """Writes a file as a series of lists"""
    file = False

    def __init__(self, filename):
        if isinstance(filename, file):
            self.file = filename
            self.filename = None
        else:
            self.filename = filename

    def write(self, strings):
        """Write a list of strings"""
        for string in strings:
            if not isinstance(string, str):
                Trace.error('Not a string: ' + str(string) + ' in ' + str(strings))
                return
            self.writestring(string)

    def writestring(self, string):
        """Write a string"""
        if not self.file:
            self.file = codecs.open(self.filename, 'w', 'utf-8')
        if self.file == sys.stdout and sys.version_info < (3, 0):
            string = string.encode('utf-8')
        self.file.write(string)

    def writeline(self, line):
        """Write a line to file"""
        self.writestring(line + '\n')

    def close(self):
        self.file.close()