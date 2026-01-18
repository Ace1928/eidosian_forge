from antlr4 import *
from io import StringIO
import sys
class IdentityTransformContext(TransformContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitIdentityTransform'):
            return visitor.visitIdentityTransform(self)
        else:
            return visitor.visitChildren(self)