from antlr4 import *
from io import StringIO
import sys
class Id_diffContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def diff(self):
        return self.getTypedRuleContext(AutolevParser.DiffContext, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_id_diff

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterId_diff'):
            listener.enterId_diff(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitId_diff'):
            listener.exitId_diff(self)