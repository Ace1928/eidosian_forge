import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaProcessor(object):
    """A processor specifically for formulas."""
    processors = []

    def process(self, bit):
        """Process the contents of every formula bit, recursively."""
        self.processcontents(bit)
        self.processinsides(bit)
        self.traversewhole(bit)

    def processcontents(self, bit):
        """Process the contents of a formula bit."""
        if not isinstance(bit, FormulaBit):
            return
        bit.process()
        for element in bit.contents:
            self.processcontents(element)

    def processinsides(self, bit):
        """Process the insides (limits, brackets) in a formula bit."""
        if not isinstance(bit, FormulaBit):
            return
        for index, element in enumerate(bit.contents):
            for processor in self.processors:
                processor.process(bit.contents, index)
            self.processinsides(element)

    def traversewhole(self, formula):
        """Traverse over the contents to alter variables and space units."""
        last = None
        for bit, contents in self.traverse(formula):
            if bit.type == 'alpha':
                self.italicize(bit, contents)
            elif bit.type == 'font' and last and (last.type == 'number'):
                bit.contents.insert(0, FormulaConstant('\u205f'))
            last = bit

    def traverse(self, bit):
        """Traverse a formula and yield a flattened structure of (bit, list) pairs."""
        for element in bit.contents:
            if hasattr(element, 'type') and element.type:
                yield (element, bit.contents)
            elif isinstance(element, FormulaBit):
                for pair in self.traverse(element):
                    yield pair

    def italicize(self, bit, contents):
        """Italicize the given bit of text."""
        index = contents.index(bit)
        contents[index] = TaggedBit().complete([bit], 'i')