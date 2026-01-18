import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TextPosition(Position):
    """A parse position based on a raw text."""

    def __init__(self, text):
        """Create the position from elyxer.some text."""
        Position.__init__(self)
        self.pos = 0
        self.text = text
        self.checkbytemark()

    def skip(self, string):
        """Skip a string of characters."""
        self.pos += len(string)

    def identifier(self):
        """Return a sample of the remaining text."""
        length = 30
        if self.pos + length > len(self.text):
            length = len(self.text) - self.pos
        return '*' + self.text[self.pos:self.pos + length] + '*'

    def isout(self):
        """Find out if we are out of the text yet."""
        return self.pos >= len(self.text)

    def current(self):
        """Return the current character, assuming we are not out."""
        return self.text[self.pos]

    def extract(self, length):
        """Extract the next string of the given length, or None if not enough text."""
        if self.pos + length > len(self.text):
            return None
        return self.text[self.pos:self.pos + length]