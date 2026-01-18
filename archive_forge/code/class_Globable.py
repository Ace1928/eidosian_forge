import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class Globable(object):
    """A bit of text which can be globbed (lumped together in bits).
  Methods current(), skipcurrent(), checkfor() and isout() have to be
  implemented by subclasses."""
    leavepending = False

    def __init__(self):
        self.endinglist = EndingList()

    def checkbytemark(self):
        """Check for a Unicode byte mark and skip it."""
        if self.finished():
            return
        if ord(self.current()) == 65279:
            self.skipcurrent()

    def isout(self):
        """Find out if we are out of the position yet."""
        Trace.error('Unimplemented isout()')
        return True

    def current(self):
        """Return the current character."""
        Trace.error('Unimplemented current()')
        return ''

    def checkfor(self, string):
        """Check for the given string in the current position."""
        Trace.error('Unimplemented checkfor()')
        return False

    def finished(self):
        """Find out if the current text has finished."""
        if self.isout():
            if not self.leavepending:
                self.endinglist.checkpending()
            return True
        return self.endinglist.checkin(self)

    def skipcurrent(self):
        """Return the current character and skip it."""
        Trace.error('Unimplemented skipcurrent()')
        return ''

    def glob(self, currentcheck):
        """Glob a bit of text that satisfies a check on the current char."""
        glob = ''
        while not self.finished() and currentcheck():
            glob += self.skipcurrent()
        return glob

    def globalpha(self):
        """Glob a bit of alpha text"""
        return self.glob(lambda: self.current().isalpha())

    def globnumber(self):
        """Glob a row of digits."""
        return self.glob(lambda: self.current().isdigit())

    def isidentifier(self):
        """Return if the current character is alphanumeric or _."""
        if self.current().isalnum() or self.current() == '_':
            return True
        return False

    def globidentifier(self):
        """Glob alphanumeric and _ symbols."""
        return self.glob(self.isidentifier)

    def isvalue(self):
        """Return if the current character is a value character:"""
        'not a bracket or a space.'
        if self.current().isspace():
            return False
        if self.current() in '{}()':
            return False
        return True

    def globvalue(self):
        """Glob a value: any symbols but brackets."""
        return self.glob(self.isvalue)

    def skipspace(self):
        """Skip all whitespace at current position."""
        return self.glob(lambda: self.current().isspace())

    def globincluding(self, magicchar):
        """Glob a bit of text up to (including) the magic char."""
        glob = self.glob(lambda: self.current() != magicchar) + magicchar
        self.skip(magicchar)
        return glob

    def globexcluding(self, excluded):
        """Glob a bit of text up until (excluding) any excluded character."""
        return self.glob(lambda: self.current() not in excluded)

    def pushending(self, ending, optional=False):
        """Push a new ending to the bottom"""
        self.endinglist.add(ending, optional)

    def popending(self, expected=None):
        """Pop the ending found at the current position"""
        if self.isout() and self.leavepending:
            return expected
        ending = self.endinglist.pop(self)
        if expected and expected != ending:
            Trace.error('Expected ending ' + expected + ', got ' + ending)
        self.skip(ending)
        return ending

    def nextending(self):
        """Return the next ending in the queue."""
        nextending = self.endinglist.findending(self)
        if not nextending:
            return None
        return nextending.ending