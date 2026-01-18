from antlr4 import CommonTokenStream, DFA, PredictionContextCache, Lexer, LexerATNSimulator, ParserRuleContext, TerminalNode
from antlr4.InputStream import InputStream
from antlr4.Parser import Parser
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import LexerNoViableAltException
from antlr4.tree.Tree import ParseTree
from antlr4.tree.Trees import Trees
from io import StringIO
from antlr4.xpath.XPathLexer import XPathLexer
class XPathElement(object):

    def __init__(self, nodeName: str):
        self.nodeName = nodeName
        self.invert = False

    def __str__(self):
        return type(self).__name__ + '[' + ('!' if self.invert else '') + self.nodeName + ']'