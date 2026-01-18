from antlr4 import *
from io import StringIO
import sys
class Eval_atContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def BAR(self):
        return self.getToken(LaTeXParser.BAR, 0)

    def eval_at_sup(self):
        return self.getTypedRuleContext(LaTeXParser.Eval_at_supContext, 0)

    def eval_at_sub(self):
        return self.getTypedRuleContext(LaTeXParser.Eval_at_subContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_eval_at