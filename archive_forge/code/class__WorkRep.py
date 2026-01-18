import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class _WorkRep(object):
    __slots__ = ('sign', 'int', 'exp')

    def __init__(self, value=None):
        if value is None:
            self.sign = None
            self.int = 0
            self.exp = None
        elif isinstance(value, Decimal):
            self.sign = value._sign
            self.int = int(value._int)
            self.exp = value._exp
        else:
            self.sign = value[0]
            self.int = value[1]
            self.exp = value[2]

    def __repr__(self):
        return '(%r, %r, %r)' % (self.sign, self.int, self.exp)