import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FilePosition(Position):
    """A parse position based on an underlying file."""

    def __init__(self, filename):
        """Create the position from a file."""
        Position.__init__(self)
        self.reader = LineReader(filename)
        self.pos = 0
        self.checkbytemark()

    def skip(self, string):
        """Skip a string of characters."""
        length = len(string)
        while self.pos + length > len(self.reader.currentline()):
            length -= len(self.reader.currentline()) - self.pos + 1
            self.nextline()
        self.pos += length

    def currentline(self):
        """Get the current line of the underlying file."""
        return self.reader.currentline()

    def nextline(self):
        """Go to the next line."""
        self.reader.nextline()
        self.pos = 0

    def linenumber(self):
        """Return the line number of the file."""
        return self.reader.linenumber + 1

    def identifier(self):
        """Return the current line and line number in the file."""
        before = self.reader.currentline()[:self.pos - 1]
        after = self.reader.currentline()[self.pos:]
        return 'line ' + str(self.getlinenumber()) + ': ' + before + '*' + after

    def isout(self):
        """Find out if we are out of the text yet."""
        if self.pos > len(self.reader.currentline()):
            if self.pos > len(self.reader.currentline()) + 1:
                Trace.error('Out of the line ' + self.reader.currentline() + ': ' + str(self.pos))
            self.nextline()
        return self.reader.finished()

    def current(self):
        """Return the current character, assuming we are not out."""
        if self.pos == len(self.reader.currentline()):
            return '\n'
        if self.pos > len(self.reader.currentline()):
            Trace.error('Out of the line ' + self.reader.currentline() + ': ' + str(self.pos))
            return '*'
        return self.reader.currentline()[self.pos]

    def extract(self, length):
        """Extract the next string of the given length, or None if not enough text."""
        if self.pos + length > len(self.reader.currentline()):
            return None
        return self.reader.currentline()[self.pos:self.pos + length]