import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_add__(self, a, b):
    if hasattr(a, '__len__') and hasattr(b, '__len__') and (len(a) + len(b) > self.__rl_max_len__):
        raise BadCode('excessive length')
    return a + b