import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LabelFunction(CommandBit):
    """A function that acts as a label"""
    commandmap = FormulaConfig.labelfunctions

    def parsebit(self, pos):
        """Parse a literal parameter"""
        self.key = self.parseliteral(pos)

    def process(self):
        """Add an anchor with the label contents."""
        self.type = 'font'
        self.label = Label().create(' ', self.key, type='eqnumber')
        self.contents = [self.label]
        Label.names[self.key] = self.label