import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
class PrecededBy(ParseElementEnhance):
    """Lookbehind matching of the given parse expression.
    ``PrecededBy`` does not advance the parsing position within the
    input string, it only verifies that the specified parse expression
    matches prior to the current position.  ``PrecededBy`` always
    returns a null token list, but if a results name is defined on the
    given expression, it is returned.

    Parameters:

     - expr - expression that must match prior to the current parse
       location
     - retreat - (default= ``None``) - (int) maximum number of characters
       to lookbehind prior to the current parse location

    If the lookbehind expression is a string, Literal, Keyword, or
    a Word or CharsNotIn with a specified exact or maximum length, then
    the retreat parameter is not required. Otherwise, retreat must be
    specified to give a maximum number of characters to look back from
    the current parse position for a lookbehind match.

    Example::

        # VB-style variable names with type prefixes
        int_var = PrecededBy("#") + pyparsing_common.identifier
        str_var = PrecededBy("$") + pyparsing_common.identifier

    """

    def __init__(self, expr, retreat=None):
        super(PrecededBy, self).__init__(expr)
        self.expr = self.expr().leaveWhitespace()
        self.mayReturnEmpty = True
        self.mayIndexError = False
        self.exact = False
        if isinstance(expr, str):
            retreat = len(expr)
            self.exact = True
        elif isinstance(expr, (Literal, Keyword)):
            retreat = expr.matchLen
            self.exact = True
        elif isinstance(expr, (Word, CharsNotIn)) and expr.maxLen != _MAX_INT:
            retreat = expr.maxLen
            self.exact = True
        elif isinstance(expr, _PositionToken):
            retreat = 0
            self.exact = True
        self.retreat = retreat
        self.errmsg = 'not preceded by ' + str(expr)
        self.skipWhitespace = False
        self.parseAction.append(lambda s, l, t: t.__delitem__(slice(None, None)))

    def parseImpl(self, instring, loc=0, doActions=True):
        if self.exact:
            if loc < self.retreat:
                raise ParseException(instring, loc, self.errmsg)
            start = loc - self.retreat
            _, ret = self.expr._parse(instring, start)
        else:
            test_expr = self.expr + StringEnd()
            instring_slice = instring[max(0, loc - self.retreat):loc]
            last_expr = ParseException(instring, loc, self.errmsg)
            for offset in range(1, min(loc, self.retreat + 1) + 1):
                try:
                    _, ret = test_expr._parse(instring_slice, len(instring_slice) - offset)
                except ParseBaseException as pbe:
                    last_expr = pbe
                else:
                    break
            else:
                raise last_expr
        return (loc, ret)