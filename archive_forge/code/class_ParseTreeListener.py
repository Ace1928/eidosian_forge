from antlr4.Token import Token
class ParseTreeListener(object):

    def visitTerminal(self, node: TerminalNode):
        pass

    def visitErrorNode(self, node: ErrorNode):
        pass

    def enterEveryRule(self, ctx: ParserRuleContext):
        pass

    def exitEveryRule(self, ctx: ParserRuleContext):
        pass