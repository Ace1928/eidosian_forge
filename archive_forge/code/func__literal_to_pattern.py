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
def _literal_to_pattern(literal):
    assert isinstance(literal, Token)
    v = literal.value
    flag_start = _rfind(v, '/"') + 1
    assert flag_start > 0
    flags = v[flag_start:]
    assert all((f in _RE_FLAGS for f in flags)), flags
    if literal.type == 'STRING' and '\n' in v:
        raise GrammarError('You cannot put newlines in string literals')
    if literal.type == 'REGEXP' and '\n' in v and ('x' not in flags):
        raise GrammarError('You can only use newlines in regular expressions with the `x` (verbose) flag')
    v = v[:flag_start]
    assert v[0] == v[-1] and v[0] in '"/'
    x = v[1:-1]
    s = eval_escaping(x)
    if s == '':
        raise GrammarError('Empty terminals are not allowed (%s)' % literal)
    if literal.type == 'STRING':
        s = s.replace('\\\\', '\\')
        return PatternStr(s, flags, raw=literal.value)
    elif literal.type == 'REGEXP':
        return PatternRE(s, flags, raw=literal.value)
    else:
        assert False, 'Invariant failed: literal.type not in ["STRING", "REGEXP"]'