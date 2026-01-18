from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import optparse
import sys
import antlr3
from six.moves import input
class WalkerMain(_Main):

    def __init__(self, walkerClass):
        _Main.__init__(self)
        self.lexerClass = None
        self.parserClass = None
        self.walkerClass = walkerClass

    def setupOptions(self, optParser):
        optParser.add_option('--lexer', action='store', type='string', dest='lexerClass', default=None)
        optParser.add_option('--parser', action='store', type='string', dest='parserClass', default=None)
        optParser.add_option('--parser-rule', action='store', type='string', dest='parserRule', default=None)
        optParser.add_option('--rule', action='store', type='string', dest='walkerRule')

    def setUp(self, options):
        lexerMod = __import__(options.lexerClass)
        self.lexerClass = getattr(lexerMod, options.lexerClass)
        parserMod = __import__(options.parserClass)
        self.parserClass = getattr(parserMod, options.parserClass)

    def parseStream(self, options, inStream):
        lexer = self.lexerClass(inStream)
        tokenStream = antlr3.CommonTokenStream(lexer)
        parser = self.parserClass(tokenStream)
        result = getattr(parser, options.parserRule)()
        if result is not None:
            assert hasattr(result, 'tree'), 'Parser did not return an AST'
            nodeStream = antlr3.tree.CommonTreeNodeStream(result.tree)
            nodeStream.setTokenStream(tokenStream)
            walker = self.walkerClass(nodeStream)
            result = getattr(walker, options.walkerRule)()
            if result is not None:
                if hasattr(result, 'tree'):
                    self.writeln(options, result.tree.toStringTree())
                else:
                    self.writeln(options, repr(result))