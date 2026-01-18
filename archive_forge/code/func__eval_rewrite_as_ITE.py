from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def _eval_rewrite_as_ITE(self, *args, **kwargs):
    byfree = {}
    args = list(args)
    default = any((c == True for b, c in args))
    for i, (b, c) in enumerate(args):
        if not isinstance(b, Boolean) and b != True:
            raise TypeError(filldedent('\n                    Expecting Boolean or bool but got `%s`\n                    ' % func_name(b)))
        if c == True:
            break
        for c in c.args if isinstance(c, Or) else [c]:
            free = c.free_symbols
            x = free.pop()
            try:
                byfree[x] = byfree.setdefault(x, S.EmptySet).union(c.as_set())
            except NotImplementedError:
                if not default:
                    raise NotImplementedError(filldedent('\n                            A method to determine whether a multivariate\n                            conditional is consistent with a complete coverage\n                            of all variables has not been implemented so the\n                            rewrite is being stopped after encountering `%s`.\n                            This error would not occur if a default expression\n                            like `(foo, True)` were given.\n                            ' % c))
            if byfree[x] in (S.UniversalSet, S.Reals):
                args[i] = list(args[i])
                c = args[i][1] = True
                break
        if c == True:
            break
    if c != True:
        raise ValueError(filldedent('\n                Conditions must cover all reals or a final default\n                condition `(foo, True)` must be given.\n                '))
    last, _ = args[i]
    for a, c in reversed(args[:i]):
        last = ITE(c, a, last)
    return _canonical(last)