import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
import pkgutil
from ast import literal_eval
from contextlib import suppress
from typing import List, Tuple, Union, Callable, Dict, Optional, Sequence, Generator
from .utils import bfs, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique, small_factors, OrderedSet
from .lexer import Token, TerminalDef, PatternStr, PatternRE, Pattern
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol, TOKEN_DEFAULT_PRIORITY
from .utils import classify, dedup_list
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError, UnexpectedInput
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
class TerminalTreeToPattern(Transformer_NonRecursive):

    def pattern(self, ps):
        p, = ps
        return p

    def expansion(self, items: List[Pattern]) -> Pattern:
        if not items:
            return PatternStr('')
        if len(items) == 1:
            return items[0]
        pattern = ''.join((i.to_regexp() for i in items))
        return _make_joined_pattern(pattern, {i.flags for i in items})

    def expansions(self, exps: List[Pattern]) -> Pattern:
        if len(exps) == 1:
            return exps[0]
        exps.sort(key=lambda x: (-x.max_width, -x.min_width, -len(x.value)))
        pattern = '(?:%s)' % '|'.join((i.to_regexp() for i in exps))
        return _make_joined_pattern(pattern, {i.flags for i in exps})

    def expr(self, args) -> Pattern:
        inner: Pattern
        inner, op = args[:2]
        if op == '~':
            if len(args) == 3:
                op = '{%d}' % int(args[2])
            else:
                mn, mx = map(int, args[2:])
                if mx < mn:
                    raise GrammarError("Bad Range for %s (%d..%d isn't allowed)" % (inner, mn, mx))
                op = '{%d,%d}' % (mn, mx)
        else:
            assert len(args) == 2
        return PatternRE('(?:%s)%s' % (inner.to_regexp(), op), inner.flags)

    def maybe(self, expr):
        return self.expr(expr + ['?'])

    def alias(self, t):
        raise GrammarError('Aliasing not allowed in terminals (You used -> in the wrong place)')

    def value(self, v):
        return v[0]