import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Combine(TokenConverter):
    """Converter to concatenate all matching tokens to a single string.
       By default, the matching patterns must also be contiguous in the input string;
       this can be disabled by specifying C{'adjacent=False'} in the constructor.
    """

    def __init__(self, expr, joinString='', adjacent=True):
        super(Combine, self).__init__(expr)
        if adjacent:
            self.leaveWhitespace()
        self.adjacent = adjacent
        self.skipWhitespace = True
        self.joinString = joinString
        self.callPreparse = True

    def ignore(self, other):
        if self.adjacent:
            ParserElement.ignore(self, other)
        else:
            super(Combine, self).ignore(other)
        return self

    def postParse(self, instring, loc, tokenlist):
        retToks = tokenlist.copy()
        del retToks[:]
        retToks += ParseResults([''.join(tokenlist._asStringList(self.joinString))], modal=self.modalResults)
        if self.resultsName and len(retToks.keys()) > 0:
            return [retToks]
        else:
            return retToks