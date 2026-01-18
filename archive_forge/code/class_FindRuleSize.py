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
class FindRuleSize(Transformer):

    def __init__(self, keep_all_tokens: bool):
        self.keep_all_tokens = keep_all_tokens

    def _will_not_get_removed(self, sym: Symbol) -> bool:
        if isinstance(sym, NonTerminal):
            return not sym.name.startswith('_')
        if isinstance(sym, Terminal):
            return self.keep_all_tokens or not sym.filter_out
        if sym is _EMPTY:
            return False
        assert False, sym

    def _args_as_int(self, args: List[Union[int, Symbol]]) -> Generator[int, None, None]:
        for a in args:
            if isinstance(a, int):
                yield a
            elif isinstance(a, Symbol):
                yield (1 if self._will_not_get_removed(a) else 0)
            else:
                assert False

    def expansion(self, args) -> int:
        return sum(self._args_as_int(args))

    def expansions(self, args) -> int:
        return max(self._args_as_int(args))