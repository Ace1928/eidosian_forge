import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class CharsNotIn(Token):
    """Token for matching words composed of characters *not* in a given set.
       Defined with string containing all disallowed characters, and an optional
       minimum, maximum, and/or exact length.  The default value for C{min} is 1 (a
       minimum value < 1 is not valid); the default values for C{max} and C{exact}
       are 0, meaning no maximum or exact length restriction.
    """

    def __init__(self, notChars, min=1, max=0, exact=0):
        super(CharsNotIn, self).__init__()
        self.skipWhitespace = False
        self.notChars = notChars
        if min < 1:
            raise ValueError('cannot specify a minimum length < 1; use Optional(CharsNotIn()) if zero-length char group is permitted')
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayReturnEmpty = self.minLen == 0
        self.mayIndexError = False

    def parseImpl(self, instring, loc, doActions=True):
        if instring[loc] in self.notChars:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        notchars = self.notChars
        maxlen = min(start + self.maxLen, len(instring))
        while loc < maxlen and instring[loc] not in notchars:
            loc += 1
        if loc - start < self.minLen:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])

    def __str__(self):
        try:
            return super(CharsNotIn, self).__str__()
        except:
            pass
        if self.strRepr is None:
            if len(self.notChars) > 4:
                self.strRepr = '!W:(%s...)' % self.notChars[:4]
            else:
                self.strRepr = '!W:(%s)' % self.notChars
        return self.strRepr