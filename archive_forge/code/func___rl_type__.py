import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_type__(self, *args):
    if len(args) == 1:
        return type(*args)
    raise BadCode('type call error')