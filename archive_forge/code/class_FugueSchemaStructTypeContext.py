from antlr4 import *
from io import StringIO
import sys
class FugueSchemaStructTypeContext(FugueSchemaTypeContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueSchema(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchemaStructType'):
            return visitor.visitFugueSchemaStructType(self)
        else:
            return visitor.visitChildren(self)