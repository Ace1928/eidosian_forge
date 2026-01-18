import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _default_fs_class(obj):
    if isinstance(obj, FeatStruct):
        return FeatStruct
    if isinstance(obj, (dict, list)):
        return (dict, list)
    else:
        raise ValueError('To unify objects of type %s, you must specify fs_class explicitly.' % obj.__class__.__name__)