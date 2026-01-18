from antlr4 import *
from io import StringIO
import sys
class TypeConstructorContext(ConstantContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTypeConstructor'):
            return visitor.visitTypeConstructor(self)
        else:
            return visitor.visitChildren(self)