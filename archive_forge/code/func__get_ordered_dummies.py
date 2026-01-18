from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def _get_ordered_dummies(mul, verbose=False):
    """Returns all dummies in the mul sorted in canonical order.

    Explanation
    ===========

    The purpose of the canonical ordering is that dummies can be substituted
    consistently across terms with the result that equivalent terms can be
    simplified.

    It is not possible to determine if two terms are equivalent based solely on
    the dummy order.  However, a consistent substitution guided by the ordered
    dummies should lead to trivially (non-)equivalent terms, thereby revealing
    the equivalence.  This also means that if two terms have identical sequences of
    dummies, the (non-)equivalence should already be apparent.

    Strategy
    --------

    The canonical order is given by an arbitrary sorting rule.  A sort key
    is determined for each dummy as a tuple that depends on all factors where
    the index is present.  The dummies are thereby sorted according to the
    contraction structure of the term, instead of sorting based solely on the
    dummy symbol itself.

    After all dummies in the term has been assigned a key, we check for identical
    keys, i.e. unorderable dummies.  If any are found, we call a specialized
    method, _determine_ambiguous(), that will determine a unique order based
    on recursive calls to _get_ordered_dummies().

    Key description
    ---------------

    A high level description of the sort key:

        1. Range of the dummy index
        2. Relation to external (non-dummy) indices
        3. Position of the index in the first factor
        4. Position of the index in the second factor

    The sort key is a tuple with the following components:

        1. A single character indicating the range of the dummy (above, below
           or general.)
        2. A list of strings with fully masked string representations of all
           factors where the dummy is present.  By masked, we mean that dummies
           are represented by a symbol to indicate either below fermi, above or
           general.  No other information is displayed about the dummies at
           this point.  The list is sorted stringwise.
        3. An integer number indicating the position of the index, in the first
           factor as sorted in 2.
        4. An integer number indicating the position of the index, in the second
           factor as sorted in 2.

    If a factor is either of type AntiSymmetricTensor or SqOperator, the index
    position in items 3 and 4 is indicated as 'upper' or 'lower' only.
    (Creation operators are considered upper and annihilation operators lower.)

    If the masked factors are identical, the two factors cannot be ordered
    unambiguously in item 2.  In this case, items 3, 4 are left out.  If several
    indices are contracted between the unorderable factors, it will be handled by
    _determine_ambiguous()


    """
    args = Mul.make_args(mul)
    fac_dum = {fac: fac.atoms(Dummy) for fac in args}
    fac_repr = {fac: __kprint(fac) for fac in args}
    all_dums = set().union(*fac_dum.values())
    mask = {}
    for d in all_dums:
        if d.assumptions0.get('below_fermi'):
            mask[d] = '0'
        elif d.assumptions0.get('above_fermi'):
            mask[d] = '1'
        else:
            mask[d] = '2'
    dum_repr = {d: __kprint(d) for d in all_dums}

    def _key(d):
        dumstruct = [fac for fac in fac_dum if d in fac_dum[fac]]
        other_dums = set().union(*[fac_dum[fac] for fac in dumstruct])
        fac = dumstruct[-1]
        if other_dums is fac_dum[fac]:
            other_dums = fac_dum[fac].copy()
        other_dums.remove(d)
        masked_facs = [fac_repr[fac] for fac in dumstruct]
        for d2 in other_dums:
            masked_facs = [fac.replace(dum_repr[d2], mask[d2]) for fac in masked_facs]
        all_masked = [fac.replace(dum_repr[d], mask[d]) for fac in masked_facs]
        masked_facs = dict(list(zip(dumstruct, masked_facs)))
        if has_dups(all_masked):
            all_masked.sort()
            return (mask[d], tuple(all_masked))
        keydict = dict(list(zip(dumstruct, all_masked)))
        dumstruct.sort(key=lambda x: keydict[x])
        all_masked.sort()
        pos_val = []
        for fac in dumstruct:
            if isinstance(fac, AntiSymmetricTensor):
                if d in fac.upper:
                    pos_val.append('u')
                if d in fac.lower:
                    pos_val.append('l')
            elif isinstance(fac, Creator):
                pos_val.append('u')
            elif isinstance(fac, Annihilator):
                pos_val.append('l')
            elif isinstance(fac, NO):
                ops = [op for op in fac if op.has(d)]
                for op in ops:
                    if isinstance(op, Creator):
                        pos_val.append('u')
                    else:
                        pos_val.append('l')
            else:
                facpos = -1
                while 1:
                    facpos = masked_facs[fac].find(dum_repr[d], facpos + 1)
                    if facpos == -1:
                        break
                    pos_val.append(facpos)
        return (mask[d], tuple(all_masked), pos_val[0], pos_val[-1])
    dumkey = dict(list(zip(all_dums, list(map(_key, all_dums)))))
    result = sorted(all_dums, key=lambda x: dumkey[x])
    if has_dups(iter(dumkey.values())):
        unordered = defaultdict(set)
        for d, k in dumkey.items():
            unordered[k].add(d)
        for k in [k for k in unordered if len(unordered[k]) < 2]:
            del unordered[k]
        unordered = [unordered[k] for k in sorted(unordered)]
        result = _determine_ambiguous(mul, result, unordered)
    return result