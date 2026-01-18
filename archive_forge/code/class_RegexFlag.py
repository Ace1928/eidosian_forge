import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class RegexFlag(enum.IntFlag):
    A = ASCII = 128
    B = BESTMATCH = 4096
    D = DEBUG = 512
    E = ENHANCEMATCH = 32768
    F = FULLCASE = 16384
    I = IGNORECASE = 2
    L = LOCALE = 4
    M = MULTILINE = 8
    P = POSIX = 65536
    R = REVERSE = 1024
    S = DOTALL = 16
    U = UNICODE = 32
    V0 = VERSION0 = 8192
    V1 = VERSION1 = 256
    W = WORD = 2048
    X = VERBOSE = 64
    T = TEMPLATE = 1

    def __repr__(self):
        if self._name_ is not None:
            return 'regex.%s' % self._name_
        value = self._value_
        members = []
        negative = value < 0
        if negative:
            value = ~value
        for m in self.__class__:
            if value & m._value_:
                value &= ~m._value_
                members.append('regex.%s' % m._name_)
        if value:
            members.append(hex(value))
        res = '|'.join(members)
        if negative:
            if len(members) > 1:
                res = '~(%s)' % res
            else:
                res = '~%s' % res
        return res
    __str__ = object.__str__