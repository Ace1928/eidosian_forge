import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class CaselessKeyword(Keyword):

    def __init__(self, matchString, identChars=Keyword.DEFAULT_KEYWORD_CHARS):
        super(CaselessKeyword, self).__init__(matchString, identChars, caseless=True)

    def parseImpl(self, instring, loc, doActions=True):
        if instring[loc:loc + self.matchLen].upper() == self.caselessmatch and (loc >= len(instring) - self.matchLen or instring[loc + self.matchLen].upper() not in self.identChars):
            return (loc + self.matchLen, self.match)
        raise ParseException(instring, loc, self.errmsg, self)