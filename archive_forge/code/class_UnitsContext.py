from antlr4 import *
from io import StringIO
import sys
class UnitsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def UnitSystem(self):
        return self.getToken(AutolevParser.UnitSystem, 0)

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def getRuleIndex(self):
        return AutolevParser.RULE_units

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterUnits'):
            listener.enterUnits(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitUnits'):
            listener.exitUnits(self)