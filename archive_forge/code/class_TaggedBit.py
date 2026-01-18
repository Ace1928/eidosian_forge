import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TaggedBit(FormulaBit):
    """A tagged string in a formula"""

    def constant(self, constant, tag):
        """Set the constant and the tag"""
        self.output = TaggedOutput().settag(tag)
        self.add(FormulaConstant(constant))
        return self

    def complete(self, contents, tag, breaklines=False):
        """Set the constant and the tag"""
        self.contents = contents
        self.output = TaggedOutput().settag(tag, breaklines)
        return self

    def selfcomplete(self, tag):
        """Set the self-closing tag, no contents (as in <hr/>)."""
        self.output = TaggedOutput().settag(tag, empty=True)
        return self