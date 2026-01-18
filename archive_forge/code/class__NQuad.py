import sys
import warnings
from functools import partial
from . import _quadpack
import numpy as np
class _NQuad:

    def __init__(self, func, ranges, opts, full_output):
        self.abserr = 0
        self.func = func
        self.ranges = ranges
        self.opts = opts
        self.maxdepth = len(ranges)
        self.full_output = full_output
        if self.full_output:
            self.out_dict = {'neval': 0}

    def integrate(self, *args, **kwargs):
        depth = kwargs.pop('depth', 0)
        if kwargs:
            raise ValueError('unexpected kwargs')
        ind = -(depth + 1)
        fn_range = self.ranges[ind]
        low, high = fn_range(*args)
        fn_opt = self.opts[ind]
        opt = dict(fn_opt(*args))
        if 'points' in opt:
            opt['points'] = [x for x in opt['points'] if low <= x <= high]
        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = partial(self.integrate, depth=depth + 1)
        quad_r = quad(f, low, high, args=args, full_output=self.full_output, **opt)
        value = quad_r[0]
        abserr = quad_r[1]
        if self.full_output:
            infodict = quad_r[2]
            if depth + 1 == self.maxdepth:
                self.out_dict['neval'] += infodict['neval']
        self.abserr = max(self.abserr, abserr)
        if depth > 0:
            return value
        elif self.full_output:
            return (value, self.abserr, self.out_dict)
        else:
            return (value, self.abserr)