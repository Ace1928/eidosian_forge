from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
class _timelex(object):
    _split_decimal = re.compile('([.,])')

    def __init__(self, instream):
        if isinstance(instream, (bytes, bytearray)):
            instream = instream.decode()
        if isinstance(instream, text_type):
            instream = StringIO(instream)
        elif getattr(instream, 'read', None) is None:
            raise TypeError('Parser must be a string or character stream, not {itype}'.format(itype=instream.__class__.__name__))
        self.instream = instream
        self.charstack = []
        self.tokenstack = []
        self.eof = False

    def get_token(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.

        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        if self.tokenstack:
            return self.tokenstack.pop(0)
        seenletters = False
        token = None
        state = None
        while not self.eof:
            if self.charstack:
                nextchar = self.charstack.pop(0)
            else:
                nextchar = self.instream.read(1)
                while nextchar == '\x00':
                    nextchar = self.instream.read(1)
            if not nextchar:
                self.eof = True
                break
            elif not state:
                token = nextchar
                if self.isword(nextchar):
                    state = 'a'
                elif self.isnum(nextchar):
                    state = '0'
                elif self.isspace(nextchar):
                    token = ' '
                    break
                else:
                    break
            elif state == 'a':
                seenletters = True
                if self.isword(nextchar):
                    token += nextchar
                elif nextchar == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break
            elif state == '0':
                if self.isnum(nextchar):
                    token += nextchar
                elif nextchar == '.' or (nextchar == ',' and len(token) >= 2):
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break
            elif state == 'a.':
                seenletters = True
                if nextchar == '.' or self.isword(nextchar):
                    token += nextchar
                elif self.isnum(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break
            elif state == '0.':
                if nextchar == '.' or self.isnum(nextchar):
                    token += nextchar
                elif self.isword(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break
        if state in ('a.', '0.') and (seenletters or token.count('.') > 1 or token[-1] in '.,'):
            l = self._split_decimal.split(token)
            token = l[0]
            for tok in l[1:]:
                if tok:
                    self.tokenstack.append(tok)
        if state == '0.' and token.count('.') == 0:
            token = token.replace(',', '.')
        return token

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token is None:
            raise StopIteration
        return token

    def next(self):
        return self.__next__()

    @classmethod
    def split(cls, s):
        return list(cls(s))

    @classmethod
    def isword(cls, nextchar):
        """ Whether or not the next character is part of a word """
        return nextchar.isalpha()

    @classmethod
    def isnum(cls, nextchar):
        """ Whether the next character is part of a number """
        return nextchar.isdigit()

    @classmethod
    def isspace(cls, nextchar):
        """ Whether the next character is whitespace """
        return nextchar.isspace()