import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_zip__(self, *args):
    return zip(*[self.__rl_args_iter__(self.__rl_getitem__(args, i)) for i in range(len(args))])