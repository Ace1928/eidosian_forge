from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _beta_rules_lines(self):
    reverse_implications = defaultdict(list)
    for n, (pre, implied) in enumerate(self.beta_rules):
        reverse_implications[implied].append((pre, n))
    yield '# Note: the order of the beta rules is used in the beta_triggers'
    yield 'beta_rules = ['
    yield ''
    m = 0
    indices = {}
    for implied in sorted(reverse_implications):
        fact, value = implied
        yield f'    # Rules implying {fact} = {value}'
        for pre, n in reverse_implications[implied]:
            indices[n] = m
            m += 1
            setstr = ', '.join(map(str, sorted(pre)))
            yield f'    ({{{setstr}}},'
            yield f'        {implied!r}),'
        yield ''
    yield '] # beta_rules'
    yield 'beta_triggers = {'
    for query in sorted(self.beta_triggers):
        fact, value = query
        triggers = [indices[n] for n in self.beta_triggers[query]]
        yield f'    {query!r}: {triggers!r},'
    yield '} # beta_triggers'