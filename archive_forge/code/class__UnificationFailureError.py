import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class _UnificationFailureError(Exception):
    """An exception that is used by ``_destructively_unify`` to abort
    unification when a failure is encountered."""