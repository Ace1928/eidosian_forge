import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class CaselessLiteral(Literal):
    """Token to match a specified string, ignoring case of letters.
       Note: the matched results will always be in the case of the given
       match string, NOT the case of the input text.
    """

    def __init__(self, matchString):
        super(CaselessLiteral, self).__init__(matchString.upper())
        self.returnString = matchString
        self.name = "'%s'" % self.returnString
        self.errmsg = 'Expected ' + self.name

    def parseImpl(self, instring, loc, doActions=True):
        if instring[loc:loc + self.matchLen].upper() == self.match:
            return (loc + self.matchLen, self.returnString)
        raise ParseException(instring, loc, self.errmsg, self)