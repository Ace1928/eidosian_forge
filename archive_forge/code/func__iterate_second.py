import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def _iterate_second(first, second, bindings, used, skipped, finalize_method, debug):
    """
    This method facilitates movement through the terms of 'other'
    """
    debug.line(f'unify({first},{second}) {bindings}')
    if not len(first) or not len(second):
        return finalize_method(first, second, bindings, used, skipped, debug)
    else:
        newskipped = (skipped[0], skipped[1] + [second[0]])
        result = _iterate_second(first, second[1:], bindings, used, newskipped, finalize_method, debug + 1)
        try:
            newbindings, newused, unused = _unify_terms(first[0], second[0], bindings, used)
            newfirst = first[1:] + skipped[0] + unused[0]
            newsecond = second[1:] + skipped[1] + unused[1]
            result += _iterate_second(newfirst, newsecond, newbindings, newused, ([], []), finalize_method, debug + 1)
        except BindingException:
            pass
        return result