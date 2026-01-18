from antlr4 import *
from io import StringIO
import sys
class NegativeOneContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterNegativeOne'):
            listener.enterNegativeOne(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitNegativeOne'):
            listener.exitNegativeOne(self)