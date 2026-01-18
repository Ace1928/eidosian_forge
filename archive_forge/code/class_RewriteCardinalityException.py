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
class RewriteCardinalityException(RuntimeError):
    """
    @brief Base class for all exceptions thrown during AST rewrite construction.

    This signifies a case where the cardinality of two or more elements
    in a subrule are different: (ID INT)+ where |ID|!=|INT|
    """

    def __init__(self, elementDescription):
        RuntimeError.__init__(self, elementDescription)
        self.elementDescription = elementDescription

    def getMessage(self):
        return self.elementDescription