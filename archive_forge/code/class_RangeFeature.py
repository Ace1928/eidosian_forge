import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class RangeFeature(Feature):
    RANGE_RE = re.compile('(-?\\d+):(-?\\d+)')

    def read_value(self, s, position, reentrances, parser):
        m = self.RANGE_RE.match(s, position)
        if not m:
            raise ValueError('range', position)
        return ((int(m.group(1)), int(m.group(2))), m.end())

    def unify_base_values(self, fval1, fval2, bindings):
        if fval1 is None:
            return fval2
        if fval2 is None:
            return fval1
        rng = (max(fval1[0], fval2[0]), min(fval1[1], fval2[1]))
        if rng[1] < rng[0]:
            return UnificationFailure
        return rng