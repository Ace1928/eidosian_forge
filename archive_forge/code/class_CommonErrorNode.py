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
class CommonErrorNode(CommonTree):
    """A node representing erroneous token range in token stream"""

    def __init__(self, input, start, stop, exc):
        CommonTree.__init__(self, None)
        if stop is None or (stop.getTokenIndex() < start.getTokenIndex() and stop.getType() != EOF):
            stop = start
        self.input = input
        self.start = start
        self.stop = stop
        self.trappedException = exc

    def isNil(self):
        return False

    def getType(self):
        return INVALID_TOKEN_TYPE

    def getText(self):
        if isinstance(self.start, Token):
            i = self.start.getTokenIndex()
            j = self.stop.getTokenIndex()
            if self.stop.getType() == EOF:
                j = self.input.size()
            badText = self.input.toString(i, j)
        elif isinstance(self.start, Tree):
            badText = self.input.toString(self.start, self.stop)
        else:
            badText = '<unknown>'
        return badText

    def toString(self):
        if isinstance(self.trappedException, MissingTokenException):
            return '<missing type: ' + str(self.trappedException.getMissingType()) + '>'
        elif isinstance(self.trappedException, UnwantedTokenException):
            return '<extraneous: ' + str(self.trappedException.getUnexpectedToken()) + ', resync=' + self.getText() + '>'
        elif isinstance(self.trappedException, MismatchedTokenException):
            return '<mismatched token: ' + str(self.trappedException.token) + ', resync=' + self.getText() + '>'
        elif isinstance(self.trappedException, NoViableAltException):
            return '<unexpected: ' + str(self.trappedException.token) + ', resync=' + self.getText() + '>'
        return '<error: ' + self.getText() + '>'