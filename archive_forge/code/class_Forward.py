import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Forward(ParseElementEnhance):
    """Forward declaration of an expression to be defined later -
       used for recursive grammars, such as algebraic infix notation.
       When the expression is known, it is assigned to the C{Forward} variable using the '<<' operator.

       Note: take care when assigning to C{Forward} not to overlook precedence of operators.
       Specifically, '|' has a lower precedence than '<<', so that::
          fwdExpr << a | b | c
       will actually be evaluated as::
          (fwdExpr << a) | b | c
       thereby leaving b and c out as parseable alternatives.  It is recommended that you
       explicitly group the values inserted into the C{Forward}::
          fwdExpr << (a | b | c)
       Converting to use the '<<=' operator instead will avoid this problem.
    """

    def __init__(self, other=None):
        super(Forward, self).__init__(other, savelist=False)

    def __ilshift__(self, other):
        if isinstance(other, basestring):
            other = ParserElement.literalStringClass(other)
        self.expr = other
        self.mayReturnEmpty = other.mayReturnEmpty
        self.strRepr = None
        self.mayIndexError = self.expr.mayIndexError
        self.mayReturnEmpty = self.expr.mayReturnEmpty
        self.setWhitespaceChars(self.expr.whiteChars)
        self.skipWhitespace = self.expr.skipWhitespace
        self.saveAsList = self.expr.saveAsList
        self.ignoreExprs.extend(self.expr.ignoreExprs)
        return self

    def __lshift__(self, other):
        warnings.warn("Operator '<<' is deprecated, use '<<=' instead", DeprecationWarning, stacklevel=2)
        self <<= other
        return None

    def leaveWhitespace(self):
        self.skipWhitespace = False
        return self

    def streamline(self):
        if not self.streamlined:
            self.streamlined = True
            if self.expr is not None:
                self.expr.streamline()
        return self

    def validate(self, validateTrace=[]):
        if self not in validateTrace:
            tmp = validateTrace[:] + [self]
            if self.expr is not None:
                self.expr.validate(tmp)
        self.checkRecursion([])

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        self._revertClass = self.__class__
        self.__class__ = _ForwardNoRecurse
        try:
            if self.expr is not None:
                retString = _ustr(self.expr)
            else:
                retString = 'None'
        finally:
            self.__class__ = self._revertClass
        return self.__class__.__name__ + ': ' + retString

    def copy(self):
        if self.expr is not None:
            return super(Forward, self).copy()
        else:
            ret = Forward()
            ret << self
            return ret