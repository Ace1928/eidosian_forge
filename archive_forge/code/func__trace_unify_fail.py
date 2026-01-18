import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _trace_unify_fail(path, result):
    if result is UnificationFailure:
        resume = ''
    else:
        resume = ' (nonfatal)'
    print('  ' + '|   ' * len(path) + '|   |')
    print('  ' + 'X   ' * len(path) + 'X   X <-- FAIL' + resume)