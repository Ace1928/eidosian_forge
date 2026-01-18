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
def _unpack_import(self, stmt, grammar_name):
    if len(stmt.children) > 1:
        path_node, arg1 = stmt.children
    else:
        path_node, = stmt.children
        arg1 = None
    if isinstance(arg1, Tree):
        dotted_path = tuple(path_node.children)
        names = arg1.children
        aliases = dict(zip(names, names))
    else:
        dotted_path = tuple(path_node.children[:-1])
        if not dotted_path:
            name, = path_node.children
            raise GrammarError('Nothing was imported from grammar `%s`' % name)
        name = path_node.children[-1]
        aliases = {name.value: (arg1 or name).value}
    if path_node.data == 'import_lib':
        base_path = None
    else:
        if grammar_name == '<string>':
            try:
                base_file = os.path.abspath(sys.modules['__main__'].__file__)
            except AttributeError:
                base_file = None
        else:
            base_file = grammar_name
        if base_file:
            if isinstance(base_file, PackageResource):
                base_path = PackageResource(base_file.pkg_name, os.path.split(base_file.path)[0])
            else:
                base_path = os.path.split(base_file)[0]
        else:
            base_path = os.path.abspath(os.path.curdir)
    return (dotted_path, base_path, aliases)