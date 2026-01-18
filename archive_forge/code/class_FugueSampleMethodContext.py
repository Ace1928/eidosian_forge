from antlr4 import *
from io import StringIO
import sys
class FugueSampleMethodContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.percentage = None
        self.rows = None

    def PERCENTLIT(self):
        return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

    def PERCENT(self):
        return self.getToken(fugue_sqlParser.PERCENT, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def APPROX(self):
        return self.getToken(fugue_sqlParser.APPROX, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_fugueSampleMethod

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSampleMethod'):
            return visitor.visitFugueSampleMethod(self)
        else:
            return visitor.visitChildren(self)