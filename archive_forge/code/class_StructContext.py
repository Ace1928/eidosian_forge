from antlr4 import *
from io import StringIO
import sys
class StructContext(PrimaryExpressionContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self._namedExpression = None
        self.argument = list()
        self.copyFrom(ctx)

    def STRUCT(self):
        return self.getToken(fugue_sqlParser.STRUCT, 0)

    def namedExpression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.NamedExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitStruct'):
            return visitor.visitStruct(self)
        else:
            return visitor.visitChildren(self)