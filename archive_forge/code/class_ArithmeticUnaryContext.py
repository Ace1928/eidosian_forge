from antlr4 import *
from io import StringIO
import sys
class ArithmeticUnaryContext(ValueExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.theOperator = None
        self.copyFrom(ctx)

    def valueExpression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def TILDE(self):
        return self.getToken(fugue_sqlParser.TILDE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitArithmeticUnary'):
            return visitor.visitArithmeticUnary(self)
        else:
            return visitor.visitChildren(self)