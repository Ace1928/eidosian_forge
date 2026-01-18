from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def POSform(variables, minterms, dontcares=None):
    """
    The POSform function uses simplified_pairs and a redundant-group
    eliminating algorithm to convert the list of all input combinations
    that generate '1' (the minterms) into the smallest product-of-sums form.

    The variables must be given as the first argument.

    Return a logical :py:class:`~.And` function (i.e., the "product of sums"
    or "POS" form) that gives the desired outcome. If there are inputs that can
    be ignored, pass them as a list, too.

    The result will be one of the (perhaps many) functions that satisfy
    the conditions.

    Examples
    ========

    >>> from sympy.logic import POSform
    >>> from sympy import symbols
    >>> w, x, y, z = symbols('w x y z')
    >>> minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
    ...             [1, 0, 1, 1], [1, 1, 1, 1]]
    >>> dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    The terms can also be represented as integers:

    >>> minterms = [1, 3, 7, 11, 15]
    >>> dontcares = [0, 2, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    z & (y | ~w)

    They can also be specified using dicts, which does not have to be fully
    specified:

    >>> minterms = [{w: 0, x: 1}, {y: 1, z: 1, x: 0}]
    >>> POSform([w, x, y, z], minterms)
    (x | y) & (x | z) & (~w | ~x)

    Or a combination:

    >>> minterms = [4, 7, 11, [1, 1, 1, 1]]
    >>> dontcares = [{w : 0, x : 0, y: 0}, 5]
    >>> POSform([w, x, y, z], minterms, dontcares)
    (w | x) & (y | ~w) & (z | ~y)

    See also
    ========

    SOPform

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Quine-McCluskey_algorithm
    .. [2] https://en.wikipedia.org/wiki/Don%27t-care_term

    """
    if not minterms:
        return false
    variables = tuple(map(sympify, variables))
    minterms = _input_to_binlist(minterms, variables)
    dontcares = _input_to_binlist(dontcares or [], variables)
    for d in dontcares:
        if d in minterms:
            raise ValueError('%s in minterms is also in dontcares' % d)
    maxterms = []
    for t in product((0, 1), repeat=len(variables)):
        t = list(t)
        if t not in minterms and t not in dontcares:
            maxterms.append(t)
    new = _simplified_pairs(maxterms + dontcares)
    essential = _rem_redundancy(new, maxterms)
    return And(*[_convert_to_varsPOS(x, variables) for x in essential])