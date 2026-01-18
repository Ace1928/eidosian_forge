from antlr4 import *
from io import StringIO
import sys
class CompContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def group(self):
        return self.getTypedRuleContext(LaTeXParser.GroupContext, 0)

    def abs_group(self):
        return self.getTypedRuleContext(LaTeXParser.Abs_groupContext, 0)

    def func(self):
        return self.getTypedRuleContext(LaTeXParser.FuncContext, 0)

    def atom(self):
        return self.getTypedRuleContext(LaTeXParser.AtomContext, 0)

    def floor(self):
        return self.getTypedRuleContext(LaTeXParser.FloorContext, 0)

    def ceil(self):
        return self.getTypedRuleContext(LaTeXParser.CeilContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_comp