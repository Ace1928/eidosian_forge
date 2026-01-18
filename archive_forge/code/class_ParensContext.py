from antlr4 import *
from io import StringIO
import sys
class ParensContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterParens'):
            listener.enterParens(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitParens'):
            listener.exitParens(self)