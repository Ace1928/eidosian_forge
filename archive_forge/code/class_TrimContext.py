from antlr4 import *
from io import StringIO
import sys
class TrimContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.trimOption = None
        self.trimStr = None
        self.srcStr = None
        self.copyFrom(ctx)

    def TRIM(self):
        return self.getToken(fugue_sqlParser.TRIM, 0)

    def FROM(self):
        return self.getToken(fugue_sqlParser.FROM, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def BOTH(self):
        return self.getToken(fugue_sqlParser.BOTH, 0)

    def LEADING(self):
        return self.getToken(fugue_sqlParser.LEADING, 0)

    def TRAILING(self):
        return self.getToken(fugue_sqlParser.TRAILING, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTrim'):
            return visitor.visitTrim(self)
        else:
            return visitor.visitChildren(self)