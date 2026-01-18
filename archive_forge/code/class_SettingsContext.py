from antlr4 import *
from io import StringIO
import sys
class SettingsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ID(self, i: int=None):
        if i is None:
            return self.getTokens(AutolevParser.ID)
        else:
            return self.getToken(AutolevParser.ID, i)

    def EXP(self):
        return self.getToken(AutolevParser.EXP, 0)

    def FLOAT(self):
        return self.getToken(AutolevParser.FLOAT, 0)

    def INT(self):
        return self.getToken(AutolevParser.INT, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_settings

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterSettings'):
            listener.enterSettings(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitSettings'):
            listener.exitSettings(self)