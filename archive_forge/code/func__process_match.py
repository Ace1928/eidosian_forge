from __future__ import annotations
from collections import deque
from dask.core import istask, subs
def _process_match(rule, syms):
    """Process a match to determine if it is correct, and to find the correct
    substitution that will convert the term into the pattern.

    Parameters
    ----------
    rule : RewriteRule
    syms : iterable
        Iterable of subterms that match a corresponding variable.

    Returns
    -------
    A dictionary of {vars : subterms} describing the substitution to make the
    pattern equivalent with the term. Returns `None` if the match is
    invalid."""
    subs = {}
    varlist = rule._varlist
    if not len(varlist) == len(syms):
        raise RuntimeError("length of varlist doesn't match length of syms.")
    for v, s in zip(varlist, syms):
        if v in subs and subs[v] != s:
            return None
        else:
            subs[v] = s
    return subs