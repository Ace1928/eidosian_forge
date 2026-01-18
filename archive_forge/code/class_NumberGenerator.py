import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class NumberGenerator(object):
    """A number generator for unique sequences and hierarchical structures. Used in:"""
    '  * ordered part numbers: Chapter 3, Section 5.3.'
    '  * unique part numbers: Footnote 15, Bibliography cite [15].'
    '  * chaptered part numbers: Figure 3.15, Equation (8.3).'
    '  * unique roman part numbers: Part I, Book IV.'
    chaptered = None
    generator = None
    romanlayouts = [x.lower() for x in NumberingConfig.layouts['roman']]
    orderedlayouts = [x.lower() for x in NumberingConfig.layouts['ordered']]
    counters = dict()
    appendix = None

    def deasterisk(self, type):
        """Remove the possible asterisk in a layout type."""
        return type.replace('*', '')

    def isunique(self, type):
        """Find out if the layout type corresponds to a unique part."""
        return self.isroman(type)

    def isroman(self, type):
        """Find out if the layout type should have roman numeration."""
        return self.deasterisk(type).lower() in self.romanlayouts

    def isinordered(self, type):
        """Find out if the layout type corresponds to an (un)ordered part."""
        return self.deasterisk(type).lower() in self.orderedlayouts

    def isnumbered(self, type):
        """Find out if the type for a layout corresponds to a numbered layout."""
        if '*' in type:
            return False
        if self.isroman(type):
            return True
        if not self.isinordered(type):
            return False
        if self.getlevel(type) > DocumentParameters.maxdepth:
            return False
        return True

    def isunordered(self, type):
        """Find out if the type contains an asterisk, basically."""
        return '*' in type

    def getlevel(self, type):
        """Get the level that corresponds to a layout type."""
        if self.isunique(type):
            return 0
        if not self.isinordered(type):
            Trace.error('Unknown layout type ' + type)
            return 0
        type = self.deasterisk(type).lower()
        level = self.orderedlayouts.index(type) + 1
        return level - DocumentParameters.startinglevel

    def getparttype(self, type):
        """Obtain the type for the part: without the asterisk, """
        'and switched to Appendix if necessary.'
        if NumberGenerator.appendix and self.getlevel(type) == 1:
            return 'Appendix'
        return self.deasterisk(type)

    def generate(self, type):
        """Generate a number for a layout type."""
        'Unique part types such as Part or Book generate roman numbers: Part I.'
        'Ordered part types return dot-separated tuples: Chapter 5, Subsection 2.3.5.'
        'Everything else generates unique numbers: Bibliography [1].'
        'Each invocation results in a new number.'
        return self.getcounter(type).getnext()

    def getcounter(self, type):
        """Get the counter for the given type."""
        type = type.lower()
        if not type in self.counters:
            self.counters[type] = self.create(type)
        return self.counters[type]

    def create(self, type):
        """Create a counter for the given type."""
        if self.isnumbered(type) and self.getlevel(type) > 1:
            index = self.orderedlayouts.index(type)
            above = self.orderedlayouts[index - 1]
            master = self.getcounter(above)
            return self.createdependent(type, master)
        counter = NumberCounter(type)
        if self.isroman(type):
            counter.setmode('I')
        return counter

    def getdependentcounter(self, type, master):
        """Get (or create) a counter of the given type that depends on another."""
        if not type in self.counters or not self.counters[type].master:
            self.counters[type] = self.createdependent(type, master)
        return self.counters[type]

    def createdependent(self, type, master):
        """Create a dependent counter given the master."""
        return DependentCounter(type).setmaster(master)

    def startappendix(self):
        """Start appendices here."""
        firsttype = self.orderedlayouts[DocumentParameters.startinglevel]
        counter = self.getcounter(firsttype)
        counter.setmode('A').reset()
        NumberGenerator.appendix = True