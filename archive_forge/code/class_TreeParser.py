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
class TreeParser(BaseRecognizer):
    """@brief Baseclass for generated tree parsers.

    A parser for a stream of tree nodes.  "tree grammars" result in a subclass
    of this.  All the error reporting and recovery is shared with Parser via
    the BaseRecognizer superclass.
    """

    def __init__(self, input, state=None):
        BaseRecognizer.__init__(self, state)
        self.input = None
        self.setTreeNodeStream(input)

    def reset(self):
        BaseRecognizer.reset(self)
        if self.input is not None:
            self.input.seek(0)

    def setTreeNodeStream(self, input):
        """Set the input stream"""
        self.input = input

    def getTreeNodeStream(self):
        return self.input

    def getSourceName(self):
        return self.input.getSourceName()

    def getCurrentInputSymbol(self, input):
        return input.LT(1)

    def getMissingSymbol(self, input, e, expectedTokenType, follow):
        tokenText = '<missing ' + self.tokenNames[expectedTokenType] + '>'
        return CommonTree(CommonToken(type=expectedTokenType, text=tokenText))

    def matchAny(self, ignore):
        """
        Match '.' in tree parser has special meaning.  Skip node or
        entire tree if node has children.  If children, scan until
        corresponding UP node.
        """
        self._state.errorRecovery = False
        look = self.input.LT(1)
        if self.input.getTreeAdaptor().getChildCount(look) == 0:
            self.input.consume()
            return
        level = 0
        tokenType = self.input.getTreeAdaptor().getType(look)
        while tokenType != EOF and (not (tokenType == UP and level == 0)):
            self.input.consume()
            look = self.input.LT(1)
            tokenType = self.input.getTreeAdaptor().getType(look)
            if tokenType == DOWN:
                level += 1
            elif tokenType == UP:
                level -= 1
        self.input.consume()

    def mismatch(self, input, ttype, follow):
        """
        We have DOWN/UP nodes in the stream that have no line info; override.
        plus we want to alter the exception type. Don't try to recover
        from tree parser errors inline...
        """
        raise MismatchedTreeNodeException(ttype, input)

    def getErrorHeader(self, e):
        """
        Prefix error message with the grammar name because message is
        always intended for the programmer because the parser built
        the input tree not the user.
        """
        return self.getGrammarFileName() + ': node from %sline %s:%s' % (['', 'after '][e.approximateLineInfo], e.line, e.charPositionInLine)

    def getErrorMessage(self, e, tokenNames):
        """
        Tree parsers parse nodes they usually have a token object as
        payload. Set the exception token and do the default behavior.
        """
        if isinstance(self, TreeParser):
            adaptor = e.input.getTreeAdaptor()
            e.token = adaptor.getToken(e.node)
            if e.token is not None:
                e.token = CommonToken(type=adaptor.getType(e.node), text=adaptor.getText(e.node))
        return BaseRecognizer.getErrorMessage(self, e, tokenNames)

    def traceIn(self, ruleName, ruleIndex):
        BaseRecognizer.traceIn(self, ruleName, ruleIndex, self.input.LT(1))

    def traceOut(self, ruleName, ruleIndex):
        BaseRecognizer.traceOut(self, ruleName, ruleIndex, self.input.LT(1))