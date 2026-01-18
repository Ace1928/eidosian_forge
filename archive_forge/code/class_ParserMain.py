from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
class ParserMain(_Main):

    def __init__(self, lexerClassName, parserClass):
        _Main.__init__(self)
        self.lexerClassName = lexerClassName
        self.lexerClass = None
        self.parserClass = parserClass

    def setupOptions(self, optParser):
        optParser.add_option('--lexer', action='store', type='string', dest='lexerClass', default=self.lexerClassName)
        optParser.add_option('--rule', action='store', type='string', dest='parserRule')

    def setUp(self, options):
        lexerMod = __import__(options.lexerClass)
        self.lexerClass = getattr(lexerMod, options.lexerClass)

    def parseStream(self, options, inStream):
        lexer = self.lexerClass(inStream)
        tokenStream = antlr3.CommonTokenStream(lexer)
        parser = self.parserClass(tokenStream)
        result = getattr(parser, options.parserRule)()
        if result is not None:
            if hasattr(result, 'tree'):
                if result.tree is not None:
                    self.writeln(options, result.tree.toStringTree())
            else:
                self.writeln(options, repr(result))