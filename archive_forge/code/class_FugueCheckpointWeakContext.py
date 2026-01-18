from antlr4 import *
from io import StringIO
import sys
class FugueCheckpointWeakContext(FugueCheckpointContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.params = None
        self.copyFrom(ctx)

    def PERSIST(self):
        return self.getToken(fugue_sqlParser.PERSIST, 0)

    def WEAK(self):
        return self.getToken(fugue_sqlParser.WEAK, 0)

    def CHECKPOINT(self):
        return self.getToken(fugue_sqlParser.CHECKPOINT, 0)

    def LAZY(self):
        return self.getToken(fugue_sqlParser.LAZY, 0)

    def fugueParams(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueCheckpointWeak'):
            return visitor.visitFugueCheckpointWeak(self)
        else:
            return visitor.visitChildren(self)