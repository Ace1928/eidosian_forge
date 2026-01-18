import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class LogicalExpressionException(Exception):

    def __init__(self, index, message):
        self.index = index
        Exception.__init__(self, message)