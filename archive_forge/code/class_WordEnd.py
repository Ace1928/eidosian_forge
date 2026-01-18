import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class WordEnd(_PositionToken):
    """Matches if the current position is at the end of a Word, and
       is not followed by any character in a given set of C{wordChars}
       (default=C{printables}). To emulate the C{\x08} behavior of regular expressions,
       use C{WordEnd(alphanums)}. C{WordEnd} will also match at the end of
       the string being parsed, or at the end of a line.
    """

    def __init__(self, wordChars=printables):
        super(WordEnd, self).__init__()
        self.wordChars = set(wordChars)
        self.skipWhitespace = False
        self.errmsg = 'Not at the end of a word'

    def parseImpl(self, instring, loc, doActions=True):
        instrlen = len(instring)
        if instrlen > 0 and loc < instrlen:
            if instring[loc] in self.wordChars or instring[loc - 1] not in self.wordChars:
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])