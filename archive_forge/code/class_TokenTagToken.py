from antlr4.Token import CommonToken
class TokenTagToken(CommonToken):
    __slots__ = ('tokenName', 'label')

    def __init__(self, tokenName: str, type: int, label: str=None):
        super().__init__(type=type)
        self.tokenName = tokenName
        self.label = label
        self._text = self.getText()

    def getText(self):
        if self.label is None:
            return '<' + self.tokenName + '>'
        else:
            return '<' + self.label + ':' + self.tokenName + '>'

    def __str__(self):
        return self.tokenName + ':' + str(self.type)