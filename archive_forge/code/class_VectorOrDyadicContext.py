from antlr4 import *
from io import StringIO
import sys
class VectorOrDyadicContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def vec(self):
        return self.getTypedRuleContext(AutolevParser.VecContext, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterVectorOrDyadic'):
            listener.enterVectorOrDyadic(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitVectorOrDyadic'):
            listener.exitVectorOrDyadic(self)