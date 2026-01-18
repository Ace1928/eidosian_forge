import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class SkipTo(ParseElementEnhance):
    """Token for skipping over all undefined text until the matched expression is found.
       If C{include} is set to true, the matched expression is also parsed (the skipped text
       and matched expression are returned as a 2-element list).  The C{ignore}
       argument is used to define grammars (typically quoted strings and comments) that
       might contain false matches.
    """

    def __init__(self, other, include=False, ignore=None, failOn=None):
        super(SkipTo, self).__init__(other)
        self.ignoreExpr = ignore
        self.mayReturnEmpty = True
        self.mayIndexError = False
        self.includeMatch = include
        self.asList = False
        if failOn is not None and isinstance(failOn, basestring):
            self.failOn = Literal(failOn)
        else:
            self.failOn = failOn
        self.errmsg = 'No match found for ' + _ustr(self.expr)

    def parseImpl(self, instring, loc, doActions=True):
        startLoc = loc
        instrlen = len(instring)
        expr = self.expr
        failParse = False
        while loc <= instrlen:
            try:
                if self.failOn:
                    try:
                        self.failOn.tryParse(instring, loc)
                    except ParseBaseException:
                        pass
                    else:
                        failParse = True
                        raise ParseException(instring, loc, 'Found expression ' + str(self.failOn))
                    failParse = False
                if self.ignoreExpr is not None:
                    while 1:
                        try:
                            loc = self.ignoreExpr.tryParse(instring, loc)
                        except ParseBaseException:
                            break
                expr._parse(instring, loc, doActions=False, callPreParse=False)
                skipText = instring[startLoc:loc]
                if self.includeMatch:
                    loc, mat = expr._parse(instring, loc, doActions, callPreParse=False)
                    if mat:
                        skipRes = ParseResults(skipText)
                        skipRes += mat
                        return (loc, [skipRes])
                    else:
                        return (loc, [skipText])
                else:
                    return (loc, [skipText])
            except (ParseException, IndexError):
                if failParse:
                    raise
                else:
                    loc += 1
        raise ParseException(instring, loc, self.errmsg, self)