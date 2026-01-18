from antlr4 import *
from io import StringIO
import sys
class UnaryContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def unary(self):
        return self.getTypedRuleContext(LaTeXParser.UnaryContext, 0)

    def ADD(self):
        return self.getToken(LaTeXParser.ADD, 0)

    def SUB(self):
        return self.getToken(LaTeXParser.SUB, 0)

    def postfix(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(LaTeXParser.PostfixContext)
        else:
            return self.getTypedRuleContext(LaTeXParser.PostfixContext, i)

    def getRuleIndex(self):
        return LaTeXParser.RULE_unary