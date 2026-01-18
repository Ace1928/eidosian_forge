import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class UnexpectedTokenException(LogicalExpressionException):

    def __init__(self, index, unexpected=None, expected=None, message=None):
        if unexpected and expected:
            msg = "Unexpected token: '%s'.  Expected token '%s'." % (unexpected, expected)
        elif unexpected:
            msg = "Unexpected token: '%s'." % unexpected
            if message:
                msg += '  ' + message
        else:
            msg = "Expected token '%s'." % expected
        LogicalExpressionException.__init__(self, index, msg)