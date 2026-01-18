import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _trace_bindings(path, bindings):
    if len(bindings) > 0:
        binditems = sorted(bindings.items(), key=lambda v: v[0].name)
        bindstr = '{%s}' % ', '.join((f'{var}: {_trace_valrepr(val)}' for var, val in binditems))
        print('  ' + '|   ' * len(path) + '    Bindings: ' + bindstr)