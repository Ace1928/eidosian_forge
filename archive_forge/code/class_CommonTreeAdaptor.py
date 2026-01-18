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
class CommonTreeAdaptor(BaseTreeAdaptor):
    """
    @brief A TreeAdaptor that works with any Tree implementation.

    It provides
    really just factory methods; all the work is done by BaseTreeAdaptor.
    If you would like to have different tokens created than ClassicToken
    objects, you need to override this and then set the parser tree adaptor to
    use your subclass.

    To get your parser to build nodes of a different type, override
    create(Token).
    """

    def dupNode(self, treeNode):
        """
        Duplicate a node.  This is part of the factory;
        override if you want another kind of node to be built.

        I could use reflection to prevent having to override this
        but reflection is slow.
        """
        if treeNode is None:
            return None
        return treeNode.dupNode()

    def createWithPayload(self, payload):
        return CommonTree(payload)

    def createToken(self, fromToken=None, tokenType=None, text=None):
        """
        Tell me how to create a token for use with imaginary token nodes.
        For example, there is probably no input symbol associated with imaginary
        token DECL, but you need to create it as a payload or whatever for
        the DECL node as in ^(DECL type ID).

        If you care what the token payload objects' type is, you should
        override this method and any other createToken variant.
        """
        if fromToken is not None:
            return CommonToken(oldToken=fromToken)
        return CommonToken(type=tokenType, text=text)

    def setTokenBoundaries(self, t, startToken, stopToken):
        """
        Track start/stop token for subtree root created for a rule.
        Only works with Tree nodes.  For rules that match nothing,
        seems like this will yield start=i and stop=i-1 in a nil node.
        Might be useful info so I'll not force to be i..i.
        """
        if t is None:
            return
        start = 0
        stop = 0
        if startToken is not None:
            start = startToken.index
        if stopToken is not None:
            stop = stopToken.index
        t.setTokenStartIndex(start)
        t.setTokenStopIndex(stop)

    def getTokenStartIndex(self, t):
        if t is None:
            return -1
        return t.getTokenStartIndex()

    def getTokenStopIndex(self, t):
        if t is None:
            return -1
        return t.getTokenStopIndex()

    def getText(self, t):
        if t is None:
            return None
        return t.getText()

    def getType(self, t):
        if t is None:
            return INVALID_TOKEN_TYPE
        return t.getType()

    def getToken(self, t):
        """
        What is the Token associated with this node?  If
        you are not using CommonTree, then you must
        override this in your own adaptor.
        """
        if isinstance(t, CommonTree):
            return t.getToken()
        return None

    def getChild(self, t, i):
        if t is None:
            return None
        return t.getChild(i)

    def getChildCount(self, t):
        if t is None:
            return 0
        return t.getChildCount()

    def getParent(self, t):
        return t.getParent()

    def setParent(self, t, parent):
        t.setParent(parent)

    def getChildIndex(self, t):
        return t.getChildIndex()

    def setChildIndex(self, t, index):
        t.setChildIndex(index)

    def replaceChildren(self, parent, startChildIndex, stopChildIndex, t):
        if parent is not None:
            parent.replaceChildren(startChildIndex, stopChildIndex, t)