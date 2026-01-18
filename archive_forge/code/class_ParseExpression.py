import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParseExpression(ParserElement):
    """Abstract subclass of ParserElement, for combining and post-processing parsed tokens."""

    def __init__(self, exprs, savelist=False):
        super(ParseExpression, self).__init__(savelist)
        if isinstance(exprs, list):
            self.exprs = exprs
        elif isinstance(exprs, basestring):
            self.exprs = [Literal(exprs)]
        else:
            try:
                self.exprs = list(exprs)
            except TypeError:
                self.exprs = [exprs]
        self.callPreparse = False

    def __getitem__(self, i):
        return self.exprs[i]

    def append(self, other):
        self.exprs.append(other)
        self.strRepr = None
        return self

    def leaveWhitespace(self):
        """Extends C{leaveWhitespace} defined in base class, and also invokes C{leaveWhitespace} on
           all contained expressions."""
        self.skipWhitespace = False
        self.exprs = [e.copy() for e in self.exprs]
        for e in self.exprs:
            e.leaveWhitespace()
        return self

    def ignore(self, other):
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                super(ParseExpression, self).ignore(other)
                for e in self.exprs:
                    e.ignore(self.ignoreExprs[-1])
        else:
            super(ParseExpression, self).ignore(other)
            for e in self.exprs:
                e.ignore(self.ignoreExprs[-1])
        return self

    def __str__(self):
        try:
            return super(ParseExpression, self).__str__()
        except:
            pass
        if self.strRepr is None:
            self.strRepr = '%s:(%s)' % (self.__class__.__name__, _ustr(self.exprs))
        return self.strRepr

    def streamline(self):
        super(ParseExpression, self).streamline()
        for e in self.exprs:
            e.streamline()
        if len(self.exprs) == 2:
            other = self.exprs[0]
            if isinstance(other, self.__class__) and (not other.parseAction) and (other.resultsName is None) and (not other.debug):
                self.exprs = other.exprs[:] + [self.exprs[1]]
                self.strRepr = None
                self.mayReturnEmpty |= other.mayReturnEmpty
                self.mayIndexError |= other.mayIndexError
            other = self.exprs[-1]
            if isinstance(other, self.__class__) and (not other.parseAction) and (other.resultsName is None) and (not other.debug):
                self.exprs = self.exprs[:-1] + other.exprs[:]
                self.strRepr = None
                self.mayReturnEmpty |= other.mayReturnEmpty
                self.mayIndexError |= other.mayIndexError
        return self

    def setResultsName(self, name, listAllMatches=False):
        ret = super(ParseExpression, self).setResultsName(name, listAllMatches)
        return ret

    def validate(self, validateTrace=[]):
        tmp = validateTrace[:] + [self]
        for e in self.exprs:
            e.validate(tmp)
        self.checkRecursion([])

    def copy(self):
        ret = super(ParseExpression, self).copy()
        ret.exprs = [e.copy() for e in self.exprs]
        return ret