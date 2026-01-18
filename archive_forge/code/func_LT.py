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
def LT(self, k):
    if self.p == -1:
        self.fillBuffer()
    if k == 0:
        return None
    if k < 0:
        return self.LB(-k)
    if self.p + k - 1 >= len(self.nodes):
        return self.eof
    return self.nodes[self.p + k - 1]