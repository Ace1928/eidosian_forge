from antlr4 import *
from io import StringIO
import sys
class ArithmeticBinaryContext(ValueExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.left = None
        self.theOperator = None
        self.right = None
        self.copyFrom(ctx)

    def valueExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ValueExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ValueExpressionContext, i)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def SLASH(self):
        return self.getToken(fugue_sqlParser.SLASH, 0)

    def PERCENT(self):
        return self.getToken(fugue_sqlParser.PERCENT, 0)

    def DIV(self):
        return self.getToken(fugue_sqlParser.DIV, 0)

    def PLUS(self):
        return self.getToken(fugue_sqlParser.PLUS, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def CONCAT_PIPE(self):
        return self.getToken(fugue_sqlParser.CONCAT_PIPE, 0)

    def AMPERSAND(self):
        return self.getToken(fugue_sqlParser.AMPERSAND, 0)

    def HAT(self):
        return self.getToken(fugue_sqlParser.HAT, 0)

    def PIPE(self):
        return self.getToken(fugue_sqlParser.PIPE, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitArithmeticBinary'):
            return visitor.visitArithmeticBinary(self)
        else:
            return visitor.visitChildren(self)