import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class IllegalTypeException(TypeException):

    def __init__(self, expression, other_type, allowed_type):
        super().__init__("Cannot set type of %s '%s' to '%s'; must match type '%s'." % (expression.__class__.__name__, expression, other_type, allowed_type))