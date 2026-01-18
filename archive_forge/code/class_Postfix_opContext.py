from antlr4 import *
from io import StringIO
import sys
class Postfix_opContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def BANG(self):
        return self.getToken(LaTeXParser.BANG, 0)

    def eval_at(self):
        return self.getTypedRuleContext(LaTeXParser.Eval_atContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_postfix_op