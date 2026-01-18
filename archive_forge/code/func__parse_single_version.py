import ast
import sys
import warnings
from typing import Iterable, Iterator, List, Set, Tuple
from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
def _parse_single_version(src: str, version: Tuple[int, int], *, type_comments: bool) -> ast.AST:
    filename = '<unknown>'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        return ast.parse(src, filename, feature_version=version, type_comments=type_comments)