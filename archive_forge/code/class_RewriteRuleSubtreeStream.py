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
class RewriteRuleSubtreeStream(RewriteRuleElementStream):
    """@brief Internal helper class."""

    def nextNode(self):
        """
        Treat next element as a single node even if it's a subtree.
        This is used instead of next() when the result has to be a
        tree root node.  Also prevents us from duplicating recently-added
        children; e.g., ^(type ID)+ adds ID to type and then 2nd iteration
        must dup the type node, but ID has been added.

        Referencing a rule result twice is ok; dup entire tree as
        we can't be adding trees as root; e.g., expr expr.

        Hideous code duplication here with super.next().  Can't think of
        a proper way to refactor.  This needs to always call dup node
        and super.next() doesn't know which to call: dup node or dup tree.
        """
        if self.dirty or (self.cursor >= len(self) and len(self) == 1):
            el = self._next()
            return self.adaptor.dupNode(el)
        el = self._next()
        return el

    def dup(self, el):
        return self.adaptor.dupTree(el)