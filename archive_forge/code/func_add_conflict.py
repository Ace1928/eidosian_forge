import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def add_conflict(fval1, fval2, path):
    conflict_list.append(path)
    return fval1