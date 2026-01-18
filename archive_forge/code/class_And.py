import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class And(ParseExpression):
    """Requires all given C{ParseExpression}s to be found in the given order.
       Expressions may be separated by whitespace.
       May be constructed using the C{'+'} operator.
    """

    class _ErrorStop(Empty):

        def __init__(self, *args, **kwargs):
            super(And._ErrorStop, self).__init__(*args, **kwargs)
            self.name = '-'
            self.leaveWhitespace()

    def __init__(self, exprs, savelist=True):
        super(And, self).__init__(exprs, savelist)
        self.mayReturnEmpty = True
        for e in self.exprs:
            if not e.mayReturnEmpty:
                self.mayReturnEmpty = False
                break
        self.setWhitespaceChars(exprs[0].whiteChars)
        self.skipWhitespace = exprs[0].skipWhitespace
        self.callPreparse = True

    def parseImpl(self, instring, loc, doActions=True):
        loc, resultlist = self.exprs[0]._parse(instring, loc, doActions, callPreParse=False)
        errorStop = False
        for e in self.exprs[1:]:
            if isinstance(e, And._ErrorStop):
                errorStop = True
                continue
            if errorStop:
                try:
                    loc, exprtokens = e._parse(instring, loc, doActions)
                except ParseSyntaxException:
                    raise
                except ParseBaseException as pe:
                    pe.__traceback__ = None
                    raise ParseSyntaxException(pe)
                except IndexError:
                    raise ParseSyntaxException(ParseException(instring, len(instring), self.errmsg, self))
            else:
                loc, exprtokens = e._parse(instring, loc, doActions)
            if exprtokens or exprtokens.keys():
                resultlist += exprtokens
        return (loc, resultlist)

    def __iadd__(self, other):
        if isinstance(other, basestring):
            other = Literal(other)
        return self.append(other)

    def checkRecursion(self, parseElementList):
        subRecCheckList = parseElementList[:] + [self]
        for e in self.exprs:
            e.checkRecursion(subRecCheckList)
            if not e.mayReturnEmpty:
                break

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + ' '.join((_ustr(e) for e in self.exprs)) + '}'
        return self.strRepr