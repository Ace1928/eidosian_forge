from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import (Assign, Comma, Call, Newline, Name,
from libfuturize.fixer_util import indentation, suitify, commatize

        a,b,c,d,e,f,*g,h,i = range(100) changes to
        _3to2list = list(range(100))
        a,b,c,d,e,f,g,h,i, = _3to2list[:6] + [_3to2list[6:-2]] + _3to2list[-2:]

        and

        for a,b,*c,d,e in iter_of_iters: do_stuff changes to
        for _3to2iter in iter_of_iters:
            _3to2list = list(_3to2iter)
            a,b,c,d,e, = _3to2list[:2] + [_3to2list[2:-2]] + _3to2list[-2:]
            do_stuff
        