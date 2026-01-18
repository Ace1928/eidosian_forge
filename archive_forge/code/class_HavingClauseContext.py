from antlr4 import *
from io import StringIO
import sys
class HavingClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def HAVING(self):
        return self.getToken(fugue_sqlParser.HAVING, 0)

    def booleanExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.BooleanExpressionContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_havingClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitHavingClause'):
            return visitor.visitHavingClause(self)
        else:
            return visitor.visitChildren(self)