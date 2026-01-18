from antlr4.Token import Token
class RuleTagToken(Token):
    __slots__ = ('label', 'ruleName')

    def __init__(self, ruleName: str, bypassTokenType: int, label: str=None):
        if ruleName is None or len(ruleName) == 0:
            raise Exception('ruleName cannot be null or empty.')
        self.source = None
        self.type = bypassTokenType
        self.channel = Token.DEFAULT_CHANNEL
        self.start = -1
        self.stop = -1
        self.tokenIndex = -1
        self.line = 0
        self.column = -1
        self.label = label
        self._text = self.getText()
        self.ruleName = ruleName

    def getText(self):
        if self.label is None:
            return '<' + self.ruleName + '>'
        else:
            return '<' + self.label + ':' + self.ruleName + '>'