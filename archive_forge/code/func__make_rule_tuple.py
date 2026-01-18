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
def _make_rule_tuple(modifiers_tree, name, params, priority_tree, expansions):
    if modifiers_tree.children:
        m, = modifiers_tree.children
        expand1 = '?' in m
        if expand1 and name.startswith('_'):
            raise GrammarError('Inlined rules (_rule) cannot use the ?rule modifier.')
        keep_all_tokens = '!' in m
    else:
        keep_all_tokens = False
        expand1 = False
    if priority_tree.children:
        p, = priority_tree.children
        priority = int(p)
    else:
        priority = None
    if params is not None:
        params = [t.value for t in params.children]
    return (name, params, expansions, RuleOptions(keep_all_tokens, expand1, priority=priority, template_source=name if params else None))