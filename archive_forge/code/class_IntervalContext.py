from antlr4 import *
from io import StringIO
import sys
class IntervalContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def INTERVAL(self):
        return self.getToken(fugue_sqlParser.INTERVAL, 0)

    def errorCapturingMultiUnitsInterval(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingMultiUnitsIntervalContext, 0)

    def errorCapturingUnitToUnitInterval(self):
        return self.getTypedRuleContext(fugue_sqlParser.ErrorCapturingUnitToUnitIntervalContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_interval

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitInterval'):
            return visitor.visitInterval(self)
        else:
            return visitor.visitChildren(self)