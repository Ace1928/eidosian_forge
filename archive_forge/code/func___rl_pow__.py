import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_pow__(self, a, b):
    try:
        if b > 0:
            if int(b * math_log10(a) + 1) > self.__rl_max_pow_digits__:
                raise BadCode
    except:
        raise BadCode('%r**%r invalid or too large' % (a, b))
    return a ** b