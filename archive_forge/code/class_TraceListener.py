import sys
from antlr4.BufferedTokenStream import TokenStream
from antlr4.CommonTokenFactory import TokenFactory
from antlr4.error.ErrorStrategy import DefaultErrorStrategy
from antlr4.InputStream import InputStream
from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.Token import Token
from antlr4.Lexer import Lexer
from antlr4.atn.ATNDeserializer import ATNDeserializer
from antlr4.atn.ATNDeserializationOptions import ATNDeserializationOptions
from antlr4.error.Errors import UnsupportedOperationException, RecognitionException
from antlr4.tree.ParseTreePatternMatcher import ParseTreePatternMatcher
from antlr4.tree.Tree import ParseTreeListener, TerminalNode, ErrorNode
class TraceListener(ParseTreeListener):
    __slots__ = '_parser'

    def __init__(self, parser):
        self._parser = parser

    def enterEveryRule(self, ctx):
        print('enter   ' + self._parser.ruleNames[ctx.getRuleIndex()] + ', LT(1)=' + self._parser._input.LT(1).text, file=self._parser._output)

    def visitTerminal(self, node):
        print('consume ' + str(node.symbol) + ' rule ' + self._parser.ruleNames[self._parser._ctx.getRuleIndex()], file=self._parser._output)

    def visitErrorNode(self, node):
        pass

    def exitEveryRule(self, ctx):
        print('exit    ' + self._parser.ruleNames[ctx.getRuleIndex()] + ', LT(1)=' + self._parser._input.LT(1).text, file=self._parser._output)