from antlr4 import *
from io import StringIO
import sys
class PositionContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.substr = None
        self.istr = None
        self.copyFrom(ctx)

    def POSITION(self):
        return self.getToken(fugue_sqlParser.POSITION, 0)

    def IN(self):
        return self.getToken(fugue_sqlParser.IN, 0)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPosition'):
            return visitor.visitPosition(self)
        else:
            return visitor.visitChildren(self)