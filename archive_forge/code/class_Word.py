import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class Word(Token):
    """Token for matching words composed of allowed character sets.
       Defined with string containing all allowed initial characters,
       an optional string containing allowed body characters (if omitted,
       defaults to the initial character set), and an optional minimum,
       maximum, and/or exact length.  The default value for C{min} is 1 (a
       minimum value < 1 is not valid); the default values for C{max} and C{exact}
       are 0, meaning no maximum or exact length restriction. An optional
       C{exclude} parameter can list characters that might be found in 
       the input C{bodyChars} string; useful to define a word of all printables
       except for one or two characters, for instance.
    """

    def __init__(self, initChars, bodyChars=None, min=1, max=0, exact=0, asKeyword=False, excludeChars=None):
        super(Word, self).__init__()
        if excludeChars:
            initChars = ''.join((c for c in initChars if c not in excludeChars))
            if bodyChars:
                bodyChars = ''.join((c for c in bodyChars if c not in excludeChars))
        self.initCharsOrig = initChars
        self.initChars = set(initChars)
        if bodyChars:
            self.bodyCharsOrig = bodyChars
            self.bodyChars = set(bodyChars)
        else:
            self.bodyCharsOrig = initChars
            self.bodyChars = set(initChars)
        self.maxSpecified = max > 0
        if min < 1:
            raise ValueError('cannot specify a minimum length < 1; use Optional(Word()) if zero-length word is permitted')
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact
        self.name = _ustr(self)
        self.errmsg = 'Expected ' + self.name
        self.mayIndexError = False
        self.asKeyword = asKeyword
        if ' ' not in self.initCharsOrig + self.bodyCharsOrig and (min == 1 and max == 0 and (exact == 0)):
            if self.bodyCharsOrig == self.initCharsOrig:
                self.reString = '[%s]+' % _escapeRegexRangeChars(self.initCharsOrig)
            elif len(self.bodyCharsOrig) == 1:
                self.reString = '%s[%s]*' % (re.escape(self.initCharsOrig), _escapeRegexRangeChars(self.bodyCharsOrig))
            else:
                self.reString = '[%s][%s]*' % (_escapeRegexRangeChars(self.initCharsOrig), _escapeRegexRangeChars(self.bodyCharsOrig))
            if self.asKeyword:
                self.reString = '\\b' + self.reString + '\\b'
            try:
                self.re = re.compile(self.reString)
            except:
                self.re = None

    def parseImpl(self, instring, loc, doActions=True):
        if self.re:
            result = self.re.match(instring, loc)
            if not result:
                raise ParseException(instring, loc, self.errmsg, self)
            loc = result.end()
            return (loc, result.group())
        if not instring[loc] in self.initChars:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        instrlen = len(instring)
        bodychars = self.bodyChars
        maxloc = start + self.maxLen
        maxloc = min(maxloc, instrlen)
        while loc < maxloc and instring[loc] in bodychars:
            loc += 1
        throwException = False
        if loc - start < self.minLen:
            throwException = True
        if self.maxSpecified and loc < instrlen and (instring[loc] in bodychars):
            throwException = True
        if self.asKeyword:
            if start > 0 and instring[start - 1] in bodychars or (loc < instrlen and instring[loc] in bodychars):
                throwException = True
        if throwException:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])

    def __str__(self):
        try:
            return super(Word, self).__str__()
        except:
            pass
        if self.strRepr is None:

            def charsAsStr(s):
                if len(s) > 4:
                    return s[:4] + '...'
                else:
                    return s
            if self.initCharsOrig != self.bodyCharsOrig:
                self.strRepr = 'W:(%s,%s)' % (charsAsStr(self.initCharsOrig), charsAsStr(self.bodyCharsOrig))
            else:
                self.strRepr = 'W:(%s)' % charsAsStr(self.initCharsOrig)
        return self.strRepr