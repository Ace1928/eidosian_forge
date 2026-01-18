from antlr4 import *
from io import StringIO
import sys
class IndexAssignContext(AssignmentContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def index(self):
        return self.getTypedRuleContext(AutolevParser.IndexContext, 0)

    def equals(self):
        return self.getTypedRuleContext(AutolevParser.EqualsContext, 0)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterIndexAssign'):
            listener.enterIndexAssign(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitIndexAssign'):
            listener.exitIndexAssign(self)