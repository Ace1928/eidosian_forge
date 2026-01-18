from antlr4 import *
from io import StringIO
import sys
class FugueSchemaMapTypeContext(FugueSchemaTypeContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def LT(self):
        return self.getToken(fugue_sqlParser.LT, 0)

    def fugueSchemaType(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaTypeContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaTypeContext, i)

    def GT(self):
        return self.getToken(fugue_sqlParser.GT, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueSchemaMapType'):
            return visitor.visitFugueSchemaMapType(self)
        else:
            return visitor.visitChildren(self)