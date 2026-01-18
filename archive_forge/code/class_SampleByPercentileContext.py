from antlr4 import *
from io import StringIO
import sys
class SampleByPercentileContext(SampleMethodContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.negativeSign = None
        self.percentage = None
        self.copyFrom(ctx)

    def PERCENTLIT(self):
        return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

    def PERCENT(self):
        return self.getToken(fugue_sqlParser.PERCENT, 0)

    def INTEGER_VALUE(self):
        return self.getToken(fugue_sqlParser.INTEGER_VALUE, 0)

    def DECIMAL_VALUE(self):
        return self.getToken(fugue_sqlParser.DECIMAL_VALUE, 0)

    def MINUS(self):
        return self.getToken(fugue_sqlParser.MINUS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSampleByPercentile'):
            return visitor.visitSampleByPercentile(self)
        else:
            return visitor.visitChildren(self)