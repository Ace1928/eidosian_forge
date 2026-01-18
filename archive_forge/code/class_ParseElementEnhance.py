import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParseElementEnhance(ParserElement):
    """Abstract subclass of C{ParserElement}, for combining and post-processing parsed tokens."""

    def __init__(self, expr, savelist=False):
        super(ParseElementEnhance, self).__init__(savelist)
        if isinstance(expr, basestring):
            expr = Literal(expr)
        self.expr = expr
        self.strRepr = None
        if expr is not None:
            self.mayIndexError = expr.mayIndexError
            self.mayReturnEmpty = expr.mayReturnEmpty
            self.setWhitespaceChars(expr.whiteChars)
            self.skipWhitespace = expr.skipWhitespace
            self.saveAsList = expr.saveAsList
            self.callPreparse = expr.callPreparse
            self.ignoreExprs.extend(expr.ignoreExprs)

    def parseImpl(self, instring, loc, doActions=True):
        if self.expr is not None:
            return self.expr._parse(instring, loc, doActions, callPreParse=False)
        else:
            raise ParseException('', loc, self.errmsg, self)

    def leaveWhitespace(self):
        self.skipWhitespace = False
        self.expr = self.expr.copy()
        if self.expr is not None:
            self.expr.leaveWhitespace()
        return self

    def ignore(self, other):
        if isinstance(other, Suppress):
            if other not in self.ignoreExprs:
                super(ParseElementEnhance, self).ignore(other)
                if self.expr is not None:
                    self.expr.ignore(self.ignoreExprs[-1])
        else:
            super(ParseElementEnhance, self).ignore(other)
            if self.expr is not None:
                self.expr.ignore(self.ignoreExprs[-1])
        return self

    def streamline(self):
        super(ParseElementEnhance, self).streamline()
        if self.expr is not None:
            self.expr.streamline()
        return self

    def checkRecursion(self, parseElementList):
        if self in parseElementList:
            raise RecursiveGrammarException(parseElementList + [self])
        subRecCheckList = parseElementList[:] + [self]
        if self.expr is not None:
            self.expr.checkRecursion(subRecCheckList)

    def validate(self, validateTrace=[]):
        tmp = validateTrace[:] + [self]
        if self.expr is not None:
            self.expr.validate(tmp)
        self.checkRecursion([])

    def __str__(self):
        try:
            return super(ParseElementEnhance, self).__str__()
        except:
            pass
        if self.strRepr is None and self.expr is not None:
            self.strRepr = '%s:(%s)' % (self.__class__.__name__, _ustr(self.expr))
        return self.strRepr