from antlr4.Token import Token
class TerminalNodeImpl(TerminalNode):
    __slots__ = ('parentCtx', 'symbol')

    def __init__(self, symbol: Token):
        self.parentCtx = None
        self.symbol = symbol

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    def getChild(self, i: int):
        return None

    def getSymbol(self):
        return self.symbol

    def getParent(self):
        return self.parentCtx

    def getPayload(self):
        return self.symbol

    def getSourceInterval(self):
        if self.symbol is None:
            return INVALID_INTERVAL
        tokenIndex = self.symbol.tokenIndex
        return (tokenIndex, tokenIndex)

    def getChildCount(self):
        return 0

    def accept(self, visitor: ParseTreeVisitor):
        return visitor.visitTerminal(self)

    def getText(self):
        return self.symbol.text

    def __str__(self):
        if self.symbol.type == Token.EOF:
            return '<EOF>'
        else:
            return self.symbol.text