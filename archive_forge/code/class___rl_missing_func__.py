import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
class __rl_missing_func__:

    def __init__(self, name):
        self.__name__ = name

    def __call__(self, *args, **kwds):
        raise BadCode('missing global %s' % self.__name__)