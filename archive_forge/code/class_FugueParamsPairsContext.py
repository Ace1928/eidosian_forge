from antlr4 import *
from io import StringIO
import sys
class FugueParamsPairsContext(FugueParamsContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.pairs = None
        self.copyFrom(ctx)

    def PARAMS(self):
        return self.getToken(fugue_sqlParser.PARAMS, 0)

    def fugueJsonPairs(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairsContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueParamsPairs'):
            return visitor.visitFugueParamsPairs(self)
        else:
            return visitor.visitChildren(self)