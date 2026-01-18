import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class _UnificationFailure:

    def __repr__(self):
        return 'nltk.featstruct.UnificationFailure'