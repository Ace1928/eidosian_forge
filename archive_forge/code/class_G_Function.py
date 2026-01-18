from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        obj = super().__new__(cls)
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        return super()._hashable_content() + self.args

    def __call__(self, z):
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        """
        Compute buckets for the fours sets of parameters.

        Explanation
        ===========

        We guarantee that any two equal Mod objects returned are actually the
        same, and that the buckets are sorted by real part (an and bq
        descendending, bm and ap ascending).

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import G_Function
        >>> from sympy.abc import y
        >>> from sympy import S

        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]
        >>> G_Function(a, b, [2], [y]).compute_buckets()
        ({0: [3, 2, 1], 1/2: [3/2]},
        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})

        """
        dicts = pan, pap, pbm, pbq = [defaultdict(list) for i in range(4)]
        for dic, lis in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)
        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items
        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))