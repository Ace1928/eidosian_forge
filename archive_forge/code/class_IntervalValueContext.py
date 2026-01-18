from antlr4 import *
from io import StringIO
import sys
class IntervalValueContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_intervalValue

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIntervalValue'):
            return visitor.visitIntervalValue(self)
        else:
            return visitor.visitChildren(self)