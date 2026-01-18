import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class InconsistentTypeHierarchyException(TypeException):

    def __init__(self, variable, expression=None):
        if expression:
            msg = "The variable '%s' was found in multiple places with different types in '%s'." % (variable, expression)
        else:
            msg = "The variable '%s' was found in multiple places with different types." % variable
        super().__init__(msg)