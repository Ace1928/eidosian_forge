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
class PrepareAnonTerminals(Transformer_InPlace):
    """Create a unique list of anonymous terminals. Attempt to give meaningful names to them when we add them"""

    def __init__(self, terminals):
        self.terminals = terminals
        self.term_set = {td.name for td in self.terminals}
        self.term_reverse = {td.pattern: td for td in terminals}
        self.i = 0
        self.rule_options = None

    @inline_args
    def pattern(self, p):
        value = p.value
        if p in self.term_reverse and p.flags != self.term_reverse[p].pattern.flags:
            raise GrammarError(u'Conflicting flags for the same terminal: %s' % p)
        term_name = None
        if isinstance(p, PatternStr):
            try:
                term_name = self.term_reverse[p].name
            except KeyError:
                try:
                    term_name = _TERMINAL_NAMES[value]
                except KeyError:
                    if value and is_id_continue(value) and is_id_start(value[0]) and (value.upper() not in self.term_set):
                        term_name = value.upper()
                if term_name in self.term_set:
                    term_name = None
        elif isinstance(p, PatternRE):
            if p in self.term_reverse:
                term_name = self.term_reverse[p].name
        else:
            assert False, p
        if term_name is None:
            term_name = '__ANON_%d' % self.i
            self.i += 1
        if term_name not in self.term_set:
            assert p not in self.term_reverse
            self.term_set.add(term_name)
            termdef = TerminalDef(term_name, p)
            self.term_reverse[p] = termdef
            self.terminals.append(termdef)
        filter_out = False if self.rule_options and self.rule_options.keep_all_tokens else isinstance(p, PatternStr)
        return Terminal(term_name, filter_out=filter_out)