import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_compile__(self, src, fname='<string>', mode='eval', flags=0, inherit=True, visit=None):
    names_seen = {}
    if not visit:
        bcode = compile(src, fname, mode=mode, flags=flags, dont_inherit=not inherit)
    else:
        astc = ast.parse(src, fname, mode)
        if eval_debug > 0:
            print('pre:\n%s\n' % astFormat(astc))
        astc = visit(astc)
        if eval_debug > 0:
            print('post:\n%s\n' % astFormat(astc))
        bcode = compile(astc, fname, mode=mode)
    return (bcode, names_seen)