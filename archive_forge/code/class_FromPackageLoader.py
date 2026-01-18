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
class FromPackageLoader:
    """
    Provides a simple way of creating custom import loaders that load from packages via ``pkgutil.get_data`` instead of using `open`.
    This allows them to be compatible even from within zip files.

    Relative imports are handled, so you can just freely use them.

    pkg_name: The name of the package. You can probably provide `__name__` most of the time
    search_paths: All the path that will be search on absolute imports.
    """
    pkg_name: str
    search_paths: Sequence[str]

    def __init__(self, pkg_name: str, search_paths: Sequence[str]=('',)) -> None:
        self.pkg_name = pkg_name
        self.search_paths = search_paths

    def __repr__(self):
        return '%s(%r, %r)' % (type(self).__name__, self.pkg_name, self.search_paths)

    def __call__(self, base_path: Union[None, str, PackageResource], grammar_path: str) -> Tuple[PackageResource, str]:
        if base_path is None:
            to_try = self.search_paths
        else:
            if not isinstance(base_path, PackageResource) or base_path.pkg_name != self.pkg_name:
                raise IOError()
            to_try = [base_path.path]
        err = None
        for path in to_try:
            full_path = os.path.join(path, grammar_path)
            try:
                text: Optional[bytes] = pkgutil.get_data(self.pkg_name, full_path)
            except IOError as e:
                err = e
                continue
            else:
                return (PackageResource(self.pkg_name, full_path), text.decode() if text else '')
        raise IOError('Cannot find grammar in given paths') from err