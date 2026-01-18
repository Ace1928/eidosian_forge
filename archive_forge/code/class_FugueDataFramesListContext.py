from antlr4 import *
from io import StringIO
import sys
class FugueDataFramesListContext(FugueDataFramesContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueDataFrame(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueDataFrameContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFrameContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFramesList'):
            return visitor.visitFugueDataFramesList(self)
        else:
            return visitor.visitChildren(self)