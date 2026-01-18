from antlr4 import *
from io import StringIO
import sys
class FloatContext(ExprContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def FLOAT(self):
        return self.getToken(AutolevParser.FLOAT, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterFloat'):
            listener.enterFloat(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitFloat'):
            listener.exitFloat(self)