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
def _define(self, name, is_term, exp, params=(), options=None, *, override=False):
    if name in self._definitions:
        if not override:
            self._grammar_error(is_term, "{Type} '{name}' defined more than once", name)
    elif override:
        self._grammar_error(is_term, 'Cannot override a nonexisting {type} {name}', name)
    if name.startswith('__'):
        self._grammar_error(is_term, 'Names starting with double-underscore are reserved (Error at {name})', name)
    self._definitions[name] = Definition(is_term, exp, params, self._check_options(is_term, options))