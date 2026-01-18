from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
def LB(self, k):
    """Look backwards k nodes"""
    if k == 0:
        return None
    if self.p - k < 0:
        return None
    return self.nodes[self.p - k]