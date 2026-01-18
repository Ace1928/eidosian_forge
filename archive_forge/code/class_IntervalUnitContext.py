from antlr4 import *
from io import StringIO
import sys
class IntervalUnitContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DAY(self):
        return self.getToken(fugue_sqlParser.DAY, 0)

    def HOUR(self):
        return self.getToken(fugue_sqlParser.HOUR, 0)

    def MINUTE(self):
        return self.getToken(fugue_sqlParser.MINUTE, 0)

    def MONTH(self):
        return self.getToken(fugue_sqlParser.MONTH, 0)

    def SECOND(self):
        return self.getToken(fugue_sqlParser.SECOND, 0)

    def YEAR(self):
        return self.getToken(fugue_sqlParser.YEAR, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_intervalUnit

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIntervalUnit'):
            return visitor.visitIntervalUnit(self)
        else:
            return visitor.visitChildren(self)