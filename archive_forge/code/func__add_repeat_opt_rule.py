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
def _add_repeat_opt_rule(self, a, b, target, target_opt, atom):
    """Creates a rule that matches atom 0 to (a*n+b)-1 times.

        When target matches n times atom, and target_opt 0 to n-1 times target_opt,

        First we generate target * i followed by target_opt, for i from 0 to a-1
        These match 0 to n*a - 1 times atom

        Then we generate target * a followed by atom * i, for i from 0 to b-1
        These match n*a to n*a + b-1 times atom

        The created rule will not have any shift/reduce conflicts so that it can be used with lalr

        Example rule when a=3, b=4:

            new_rule: target_opt
                    | target target_opt
                    | target target target_opt

                    | target target target
                    | target target target atom
                    | target target target atom atom
                    | target target target atom atom atom

        """
    key = (a, b, target, atom, 'opt')
    try:
        return self.rules_cache[key]
    except KeyError:
        new_name = self._name_rule('repeat_a%d_b%d_opt' % (a, b))
        tree = ST('expansions', [ST('expansion', [target] * i + [target_opt]) for i in range(a)] + [ST('expansion', [target] * a + [atom] * i) for i in range(b)])
        return self._add_rule(key, new_name, tree)