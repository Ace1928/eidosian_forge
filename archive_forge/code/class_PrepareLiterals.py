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
@inline_args
class PrepareLiterals(Transformer_InPlace):

    def literal(self, literal):
        return ST('pattern', [_literal_to_pattern(literal)])

    def range(self, start, end):
        assert start.type == end.type == 'STRING'
        start = start.value[1:-1]
        end = end.value[1:-1]
        assert len(eval_escaping(start)) == len(eval_escaping(end)) == 1
        regexp = '[%s-%s]' % (start, end)
        return ST('pattern', [PatternRE(regexp)])