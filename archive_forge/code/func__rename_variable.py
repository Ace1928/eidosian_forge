import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _rename_variable(var, used_vars):
    name, n = (re.sub('\\d+$', '', var.name), 2)
    if not name:
        name = '?'
    while Variable(f'{name}{n}') in used_vars:
        n += 1
    return Variable(f'{name}{n}')