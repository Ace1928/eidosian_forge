from antlr4 import *
from io import StringIO
import sys
class GroupContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def L_PAREN(self):
        return self.getToken(LaTeXParser.L_PAREN, 0)

    def expr(self):
        return self.getTypedRuleContext(LaTeXParser.ExprContext, 0)

    def R_PAREN(self):
        return self.getToken(LaTeXParser.R_PAREN, 0)

    def L_BRACKET(self):
        return self.getToken(LaTeXParser.L_BRACKET, 0)

    def R_BRACKET(self):
        return self.getToken(LaTeXParser.R_BRACKET, 0)

    def L_BRACE(self):
        return self.getToken(LaTeXParser.L_BRACE, 0)

    def R_BRACE(self):
        return self.getToken(LaTeXParser.R_BRACE, 0)

    def L_BRACE_LITERAL(self):
        return self.getToken(LaTeXParser.L_BRACE_LITERAL, 0)

    def R_BRACE_LITERAL(self):
        return self.getToken(LaTeXParser.R_BRACE_LITERAL, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_group