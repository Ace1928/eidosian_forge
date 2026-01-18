from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Recognizer import Recognizer
class LexerNoViableAltException(RecognitionException):

    def __init__(self, lexer: Lexer, input: InputStream, startIndex: int, deadEndConfigs: ATNConfigSet):
        super().__init__(message=None, recognizer=lexer, input=input, ctx=None)
        self.startIndex = startIndex
        self.deadEndConfigs = deadEndConfigs

    def __str__(self):
        symbol = ''
        if self.startIndex >= 0 and self.startIndex < self.input.size:
            symbol = self.input.getText(self.startIndex, self.startIndex)
        return "LexerNoViableAltException('" + symbol + "')"