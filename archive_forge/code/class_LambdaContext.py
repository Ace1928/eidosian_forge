from antlr4 import *
from io import StringIO
import sys
class LambdaContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def identifier(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, i)

    def expression(self):
        return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitLambda'):
            return visitor.visitLambda(self)
        else:
            return visitor.visitChildren(self)