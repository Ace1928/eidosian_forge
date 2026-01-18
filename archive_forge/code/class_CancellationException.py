from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Recognizer import Recognizer
class CancellationException(IllegalStateException):

    def __init__(self, msg: str):
        super().__init__(msg)