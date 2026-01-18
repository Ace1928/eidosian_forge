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
def addNavigationNode(self, ttype):
    """
        As we flatten the tree, we use UP, DOWN nodes to represent
        the tree structure.  When debugging we need unique nodes
        so instantiate new ones when uniqueNavigationNodes is true.
        """
    navNode = None
    if ttype == DOWN:
        if self.hasUniqueNavigationNodes():
            navNode = self.adaptor.createFromType(DOWN, 'DOWN')
        else:
            navNode = self.down
    elif self.hasUniqueNavigationNodes():
        navNode = self.adaptor.createFromType(UP, 'UP')
    else:
        navNode = self.up
    self.nodes.append(navNode)