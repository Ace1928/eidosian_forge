import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TaggedText(Container):
    """Text inside a tag"""
    output = None

    def __init__(self):
        self.parser = TextParser(self)
        self.output = TaggedOutput()

    def complete(self, contents, tag, breaklines=False):
        """Complete the tagged text and return it"""
        self.contents = contents
        self.output.tag = tag
        self.output.breaklines = breaklines
        return self

    def constant(self, text, tag, breaklines=False):
        """Complete the tagged text with a constant"""
        constant = Constant(text)
        return self.complete([constant], tag, breaklines)

    def __unicode__(self):
        """Return a printable representation."""
        if not hasattr(self.output, 'tag'):
            return 'Emtpy tagged text'
        if not self.output.tag:
            return 'Tagged <unknown tag>'
        return 'Tagged <' + self.output.tag + '>'