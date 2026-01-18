from antlr4 import *
from io import StringIO
import sys
class UnitToUnitIntervalContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.value = None
        self.ifrom = None
        self.to = None

    def TO(self):
        return self.getToken(fugue_sqlParser.TO, 0)

    def intervalValue(self):
        return self.getTypedRuleContext(fugue_sqlParser.IntervalValueContext, 0)

    def intervalUnit(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IntervalUnitContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IntervalUnitContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_unitToUnitInterval

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitUnitToUnitInterval'):
            return visitor.visitUnitToUnitInterval(self)
        else:
            return visitor.visitChildren(self)