from antlr4 import *
from io import StringIO
import sys
class RegularAssignContext(AssignmentContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ID(self):
        return self.getToken(AutolevParser.ID, 0)

    def equals(self):
        return self.getTypedRuleContext(AutolevParser.EqualsContext, 0)

    def expr(self):
        return self.getTypedRuleContext(AutolevParser.ExprContext, 0)

    def diff(self):
        return self.getTypedRuleContext(AutolevParser.DiffContext, 0)

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterRegularAssign'):
            listener.enterRegularAssign(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitRegularAssign'):
            listener.exitRegularAssign(self)