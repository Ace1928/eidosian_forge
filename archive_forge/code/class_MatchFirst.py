import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class MatchFirst(ParseExpression):
    """Requires that at least one C{ParseExpression} is found.
       If two expressions match, the first one listed is the one that will match.
       May be constructed using the C{'|'} operator.
    """

    def __init__(self, exprs, savelist=False):
        super(MatchFirst, self).__init__(exprs, savelist)
        if exprs:
            self.mayReturnEmpty = False
            for e in self.exprs:
                if e.mayReturnEmpty:
                    self.mayReturnEmpty = True
                    break
        else:
            self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        maxExcLoc = -1
        maxException = None
        for e in self.exprs:
            try:
                ret = e._parse(instring, loc, doActions)
                return ret
            except ParseException as err:
                if err.loc > maxExcLoc:
                    maxException = err
                    maxExcLoc = err.loc
            except IndexError:
                if len(instring) > maxExcLoc:
                    maxException = ParseException(instring, len(instring), e.errmsg, self)
                    maxExcLoc = len(instring)
        else:
            if maxException is not None:
                raise maxException
            else:
                raise ParseException(instring, loc, 'no defined alternatives to match', self)

    def __ior__(self, other):
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        return self.append(other)

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' | '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr

    def checkRecursion(self, parseElementList):
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)