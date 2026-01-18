from antlr4.Token import Token
class ErrorNodeImpl(TerminalNodeImpl, ErrorNode):

    def __init__(self, token: Token):
        super().__init__(token)

    def accept(self, visitor: ParseTreeVisitor):
        return visitor.visitErrorNode(self)