import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class WordStart(_PositionToken):
    """Matches if the current position is at the beginning of a Word, and
       is not preceded by any character in a given set of C{wordChars}
       (default=C{printables}). To emulate the C{\x08} behavior of regular expressions,
       use C{WordStart(alphanums)}. C{WordStart} will also match at the beginning of
       the string being parsed, or at the beginning of a line.
    """

    def __init__(self, wordChars=printables):
        super(WordStart, self).__init__()
        self.wordChars = set(wordChars)
        self.errmsg = 'Not at the start of a word'

    def parseImpl(self, instring, loc, doActions=True):
        if loc != 0:
            if instring[loc - 1] in self.wordChars or instring[loc] not in self.wordChars:
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])