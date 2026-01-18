import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ZeroOrMore(ParseElementEnhance):
    """Optional repetition of zero or more of the given expression."""

    def __init__(self, expr):
        super(ZeroOrMore, self).__init__(expr)
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        tokens = []
        try:
            loc, tokens = self.expr._parse(instring, loc, doActions, callPreParse=False)
            hasIgnoreExprs = len(self.ignoreExprs) > 0
            while 1:
                if hasIgnoreExprs:
                    preloc = self._skipIgnorables(instring, loc)
                else:
                    preloc = loc
                loc, tmptokens = self.expr._parse(instring, preloc, doActions)
                if tmptokens or tmptokens.keys():
                    tokens += tmptokens
        except (ParseException, IndexError):
            pass
        return (loc, tokens)

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '[' + _ustr(self.expr) + ']...'
        return self.strRepr

    def setResultsName(self, name, listAllMatches=False):
        ret = super(ZeroOrMore, self).setResultsName(name, listAllMatches)
        ret.saveAsList = True
        return ret