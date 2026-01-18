from antlr4 import *
from io import StringIO
import sys
class FugueDataFrameNestedContext(FugueDataFrameContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.task = None
        self.copyFrom(ctx)

    def fugueNestableTask(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueNestableTaskContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFrameNested'):
            return visitor.visitFugueDataFrameNested(self)
        else:
            return visitor.visitChildren(self)