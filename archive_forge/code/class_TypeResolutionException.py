import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class TypeResolutionException(TypeException):

    def __init__(self, expression, other_type):
        super().__init__("The type of '%s', '%s', cannot be resolved with type '%s'" % (expression, expression.type, other_type))