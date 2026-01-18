from antlr4 import *
from io import StringIO
import sys
class PrimitiveDataTypeContext(DataTypeContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def INTEGER_VALUE(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.INTEGER_VALUE)
        else:
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitPrimitiveDataType'):
            return visitor.visitPrimitiveDataType(self)
        else:
            return visitor.visitChildren(self)