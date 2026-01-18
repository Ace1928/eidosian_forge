from antlr4 import *
from io import StringIO
import sys
class ComparisonEqualOperatorContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def DOUBLEEQUAL(self):
        return self.getToken(fugue_sqlParser.DOUBLEEQUAL, 0)

    def EQUAL(self):
        return self.getToken(fugue_sqlParser.EQUAL, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_comparisonEqualOperator

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComparisonEqualOperator'):
            return visitor.visitComparisonEqualOperator(self)
        else:
            return visitor.visitChildren(self)