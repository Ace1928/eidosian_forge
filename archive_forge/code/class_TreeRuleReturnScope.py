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
class TreeRuleReturnScope(RuleReturnScope):
    """
    This is identical to the ParserRuleReturnScope except that
    the start property is a tree nodes not Token object
    when you are parsing trees.  To be generic the tree node types
    have to be Object.
    """

    def __init__(self):
        self.start = None
        self.tree = None

    def getStart(self):
        return self.start

    def getTree(self):
        return self.tree