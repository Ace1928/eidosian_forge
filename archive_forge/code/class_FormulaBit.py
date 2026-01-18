import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaBit(Container):
    """A bit of a formula"""
    type = None
    size = 1
    original = ''

    def __init__(self):
        """The formula bit type can be 'alpha', 'number', 'font'."""
        self.contents = []
        self.output = ContentsOutput()

    def setfactory(self, factory):
        """Set the internal formula factory."""
        self.factory = factory
        return self

    def add(self, bit):
        """Add any kind of formula bit already processed"""
        self.contents.append(bit)
        self.original += bit.original
        bit.parent = self

    def skiporiginal(self, string, pos):
        """Skip a string and add it to the original formula"""
        self.original += string
        if not pos.checkskip(string):
            Trace.error('String ' + string + ' not at ' + pos.identifier())

    def computesize(self):
        """Compute the size of the bit as the max of the sizes of all contents."""
        if len(self.contents) == 0:
            return 1
        self.size = max([element.size for element in self.contents])
        return self.size

    def clone(self):
        """Return a copy of itself."""
        return self.factory.parseformula(self.original)

    def __unicode__(self):
        """Get a string representation"""
        return self.__class__.__name__ + ' read in ' + self.original