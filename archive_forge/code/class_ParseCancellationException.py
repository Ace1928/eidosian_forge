from antlr4.InputStream import InputStream
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Recognizer import Recognizer
class ParseCancellationException(CancellationException):
    pass