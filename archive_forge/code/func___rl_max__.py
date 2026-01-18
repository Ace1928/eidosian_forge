import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_max__(self, arg, *args, **kwds):
    if args:
        arg = [arg]
        arg.extend(args)
    return max(self.__rl_args_iter__(arg), **kwds)