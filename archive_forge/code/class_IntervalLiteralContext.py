from antlr4 import *
from io import StringIO
import sys
class IntervalLiteralContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def interval(self):
        return self.getTypedRuleContext(fugue_sqlParser.IntervalContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIntervalLiteral'):
            return visitor.visitIntervalLiteral(self)
        else:
            return visitor.visitChildren(self)