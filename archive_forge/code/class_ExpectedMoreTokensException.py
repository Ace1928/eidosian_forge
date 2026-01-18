import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class ExpectedMoreTokensException(LogicalExpressionException):

    def __init__(self, index, message=None):
        if not message:
            message = 'More tokens expected.'
        LogicalExpressionException.__init__(self, index, 'End of input found.  ' + message)