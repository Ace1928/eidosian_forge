from antlr4 import *
from io import StringIO
import sys
class WhenClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.condition = None
        self.result = None

    def WHEN(self):
        return self.getToken(fugue_sqlParser.WHEN, 0)

    def THEN(self):
        return self.getToken(fugue_sqlParser.THEN, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_whenClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitWhenClause'):
            return visitor.visitWhenClause(self)
        else:
            return visitor.visitChildren(self)