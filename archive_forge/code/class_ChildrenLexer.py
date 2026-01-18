import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
class ChildrenLexer:

    def __init__(self, children):
        self.children = children

    def lex(self, parser_state):
        return self.children