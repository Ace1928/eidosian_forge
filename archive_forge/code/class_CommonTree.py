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
class CommonTree(BaseTree):
    """@brief A tree node that is wrapper for a Token object.

    After 3.0 release
    while building tree rewrite stuff, it became clear that computing
    parent and child index is very difficult and cumbersome.  Better to
    spend the space in every tree node.  If you don't want these extra
    fields, it's easy to cut them out in your own BaseTree subclass.

    """

    def __init__(self, payload):
        BaseTree.__init__(self)
        self.startIndex = -1
        self.stopIndex = -1
        self.parent = None
        self.childIndex = -1
        if payload is None:
            self.token = None
        elif isinstance(payload, CommonTree):
            self.token = payload.token
            self.startIndex = payload.startIndex
            self.stopIndex = payload.stopIndex
        elif payload is None or isinstance(payload, Token):
            self.token = payload
        else:
            raise TypeError(type(payload).__name__)

    def getToken(self):
        return self.token

    def dupNode(self):
        return CommonTree(self)

    def isNil(self):
        return self.token is None

    def getType(self):
        if self.token is None:
            return INVALID_TOKEN_TYPE
        return self.token.getType()
    type = property(getType)

    def getText(self):
        if self.token is None:
            return None
        return self.token.text
    text = property(getText)

    def getLine(self):
        if self.token is None or self.token.getLine() == 0:
            if self.getChildCount():
                return self.getChild(0).getLine()
            else:
                return 0
        return self.token.getLine()
    line = property(getLine)

    def getCharPositionInLine(self):
        if self.token is None or self.token.getCharPositionInLine() == -1:
            if self.getChildCount():
                return self.getChild(0).getCharPositionInLine()
            else:
                return 0
        else:
            return self.token.getCharPositionInLine()
    charPositionInLine = property(getCharPositionInLine)

    def getTokenStartIndex(self):
        if self.startIndex == -1 and self.token is not None:
            return self.token.getTokenIndex()
        return self.startIndex

    def setTokenStartIndex(self, index):
        self.startIndex = index
    tokenStartIndex = property(getTokenStartIndex, setTokenStartIndex)

    def getTokenStopIndex(self):
        if self.stopIndex == -1 and self.token is not None:
            return self.token.getTokenIndex()
        return self.stopIndex

    def setTokenStopIndex(self, index):
        self.stopIndex = index
    tokenStopIndex = property(getTokenStopIndex, setTokenStopIndex)

    def getChildIndex(self):
        return self.childIndex

    def setChildIndex(self, idx):
        self.childIndex = idx

    def getParent(self):
        return self.parent

    def setParent(self, t):
        self.parent = t

    def toString(self):
        if self.isNil():
            return 'nil'
        if self.getType() == INVALID_TOKEN_TYPE:
            return '<errornode>'
        return self.token.text
    __str__ = toString

    def toStringTree(self):
        if not self.children:
            return self.toString()
        ret = ''
        if not self.isNil():
            ret += '(%s ' % self.toString()
        ret += ' '.join([child.toStringTree() for child in self.children])
        if not self.isNil():
            ret += ')'
        return ret