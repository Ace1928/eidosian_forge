from antlr4 import *
from io import StringIO
import sys
class Func_arg_noparensContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def mp_nofunc(self):
        return self.getTypedRuleContext(LaTeXParser.Mp_nofuncContext, 0)

    def getRuleIndex(self):
        return LaTeXParser.RULE_func_arg_noparens