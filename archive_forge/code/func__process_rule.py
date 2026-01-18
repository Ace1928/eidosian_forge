from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _process_rule(self, a, b):
    if isinstance(b, And):
        sorted_bargs = sorted(b.args, key=str)
        for barg in sorted_bargs:
            self.process_rule(a, barg)
    elif isinstance(b, Or):
        sorted_bargs = sorted(b.args, key=str)
        if not isinstance(a, Logic):
            if a in sorted_bargs:
                raise TautologyDetected(a, b, 'a -> a|c|...')
        self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))
        for bidx in range(len(sorted_bargs)):
            barg = sorted_bargs[bidx]
            brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
            self.process_rule(And(a, Not(barg)), Or(*brest))
    elif isinstance(a, And):
        sorted_aargs = sorted(a.args, key=str)
        if b in sorted_aargs:
            raise TautologyDetected(a, b, 'a & b -> a')
        self.proved_rules.append((a, b))
    elif isinstance(a, Or):
        sorted_aargs = sorted(a.args, key=str)
        if b in sorted_aargs:
            raise TautologyDetected(a, b, 'a | b -> a')
        for aarg in sorted_aargs:
            self.process_rule(aarg, b)
    else:
        self.proved_rules.append((a, b))
        self.proved_rules.append((Not(b), Not(a)))