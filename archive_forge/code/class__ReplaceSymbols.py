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
class _ReplaceSymbols(Transformer_InPlace):
    """Helper for ApplyTemplates"""

    def __init__(self):
        self.names = {}

    def value(self, c):
        if len(c) == 1 and isinstance(c[0], Symbol) and (c[0].name in self.names):
            return self.names[c[0].name]
        return self.__default__('value', c, None)

    def template_usage(self, c):
        name = c[0].name
        if name in self.names:
            return self.__default__('template_usage', [self.names[name]] + c[1:], None)
        return self.__default__('template_usage', c, None)