from antlr4 import *
from io import StringIO
import sys
class FugueDataFramesDictContext(FugueDataFramesContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def fugueDataFramePair(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.FugueDataFramePairContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramePairContext, i)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueDataFramesDict'):
            return visitor.visitFugueDataFramesDict(self)
        else:
            return visitor.visitChildren(self)