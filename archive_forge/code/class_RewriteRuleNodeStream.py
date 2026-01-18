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
class RewriteRuleNodeStream(RewriteRuleElementStream):
    """
    Queues up nodes matched on left side of -> in a tree parser. This is
    the analog of RewriteRuleTokenStream for normal parsers.
    """

    def nextNode(self):
        return self._next()

    def toTree(self, el):
        return self.adaptor.dupNode(el)

    def dup(self, el):
        raise TypeError("dup can't be called for a node stream.")