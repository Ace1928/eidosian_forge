import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def _order_ref_strings(self, refs):
    strings = ['%s' % ref for ref in refs]
    ind_vars = []
    func_vars = []
    event_vars = []
    other_vars = []
    for s in strings:
        if is_indvar(s):
            ind_vars.append(s)
        elif is_funcvar(s):
            func_vars.append(s)
        elif is_eventvar(s):
            event_vars.append(s)
        else:
            other_vars.append(s)
    return sorted(other_vars) + sorted(event_vars, key=lambda v: int([v[2:], -1][len(v[2:]) == 0])) + sorted(func_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0]))) + sorted(ind_vars, key=lambda v: (v[0], int([v[1:], -1][len(v[1:]) == 0])))