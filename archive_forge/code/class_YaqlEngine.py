import collections
import re
import uuid
from ply import lex
from ply import yacc
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import lexer
from yaql.language import parser
from yaql.language import utils
class YaqlEngine(object):

    def __init__(self, ply_lexer, ply_parser, options, factory):
        self._lexer = ply_lexer
        self._parser = ply_parser
        self._options = utils.FrozenDict(options or {})
        self._factory = factory

    @property
    def lexer(self):
        return self._lexer

    @property
    def parser(self):
        return self._parser

    @property
    def options(self):
        return self._options

    @property
    def factory(self):
        return self._factory

    def __call__(self, expression, options=None):
        if options:
            return self.copy(options)(expression)
        return expressions.Statement(self.parser.parse(expression, lexer=self.lexer), self)

    def copy(self, options):
        opt = dict(self._options)
        opt.update(options)
        return YaqlEngine(self._lexer, self._parser, opt, self._factory)