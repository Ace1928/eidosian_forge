import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
class _MakeTreeMatch:

    def __init__(self, name, expansion):
        self.name = name
        self.expansion = expansion

    def __call__(self, args):
        t = Tree(self.name, args)
        t.meta.match_tree = True
        t.meta.orig_expansion = self.expansion
        return t