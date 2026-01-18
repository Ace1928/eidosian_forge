import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Regex(Token):
    """Token for matching strings that match a given regular expression.
       Defined with string specifying the regular expression in a form recognized by the inbuilt Python re module.
    """
    compiledREtype = type(re.compile('[A-Z]'))

    def __init__(self, pattern, flags=0):
        """The parameters C{pattern} and C{flags} are passed to the C{re.compile()} function as-is. See the Python C{re} module for an explanation of the acceptable patterns and flags."""
        super(Regex, self).__init__()
        if isinstance(pattern, basestring):
            if len(pattern) == 0:
                warnings.warn('null string passed to Regex; use Empty() instead', SyntaxWarning, stacklevel=2)
            self.pattern = pattern
            self.flags = flags
            try:
                self.re = re.compile(self.pattern, self.flags)
                self.reString = self.pattern
            except sre_constants.error:
                warnings.warn('invalid pattern (%s) passed to Regex' % pattern, SyntaxWarning, stacklevel=2)
                raise
        elif isinstance(pattern, Regex.compiledREtype):
            self.re = pattern
            self.pattern = self.reString = str(pattern)
            self.flags = flags
        else:
            raise ValueError('Regex may only be constructed with a string or a compiled RE object')
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        result = self.re.match(instring, loc)
        if not result:
            raise ParseException(instring, loc, self.errmsg, self)
        loc = result.end()
        d = result.groupdict()
        ret = ParseResults(result.group())
        if d:
            for k in d:
                ret[k] = d[k]
        return (loc, ret)

    def __str__(self):
        try:
            return super(Regex, self).__str__()
        except:
            pass
        if self.strRepr is None:
            self.strRepr = 'Re:(%s)' % repr(self.pattern)
        return self.strRepr