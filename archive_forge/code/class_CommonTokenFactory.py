from antlr4.Token import CommonToken
class CommonTokenFactory(TokenFactory):
    __slots__ = 'copyText'
    DEFAULT = None

    def __init__(self, copyText: bool=False):
        self.copyText = copyText

    def create(self, source, type: int, text: str, channel: int, start: int, stop: int, line: int, column: int):
        t = CommonToken(source, type, channel, start, stop)
        t.line = line
        t.column = column
        if text is not None:
            t.text = text
        elif self.copyText and source[1] is not None:
            t.text = source[1].getText(start, stop)
        return t

    def createThin(self, type: int, text: str):
        t = CommonToken(type=type)
        t.text = text
        return t