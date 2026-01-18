from antlr4 import *
from io import StringIO
import sys
class ApplyTransformContext(TransformContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.transformName = None
        self._transformArgument = None
        self.argument = list()
        self.copyFrom(ctx)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def transformArgument(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TransformArgumentContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TransformArgumentContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitApplyTransform'):
            return visitor.visitApplyTransform(self)
        else:
            return visitor.visitChildren(self)