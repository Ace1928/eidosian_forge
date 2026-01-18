from antlr4 import *
from io import StringIO
import sys
class SearchedCaseContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.elseExpression = None
        self.copyFrom(ctx)

    def CASE(self):
        return self.getToken(fugue_sqlParser.CASE, 0)

    def END(self):
        return self.getToken(fugue_sqlParser.END, 0)

    def whenClause(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.WhenClauseContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.WhenClauseContext, i)

    def ELSE(self):
        return self.getToken(fugue_sqlParser.ELSE, 0)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSearchedCase'):
            return visitor.visitSearchedCase(self)
        else:
            return visitor.visitChildren(self)