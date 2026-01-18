from antlr4 import *
from io import StringIO
import sys
class FugueParamsObjContext(FugueParamsContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.obj = None
        self.copyFrom(ctx)

    def fugueJsonObj(self):
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonObjContext, 0)

    def PARAMS(self):
        return self.getToken(fugue_sqlParser.PARAMS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitFugueParamsObj'):
            return visitor.visitFugueParamsObj(self)
        else:
            return visitor.visitChildren(self)