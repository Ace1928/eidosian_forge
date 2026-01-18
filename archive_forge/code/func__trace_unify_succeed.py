import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _trace_unify_succeed(path, fval1):
    print('  ' + '|   ' * len(path) + '|')
    print('  ' + '|   ' * len(path) + '+-->' + repr(fval1))