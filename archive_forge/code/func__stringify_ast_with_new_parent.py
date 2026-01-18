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
def _stringify_ast_with_new_parent(node: ast.AST, parent_stack: List[ast.AST], new_parent: ast.AST) -> Iterator[str]:
    parent_stack.append(new_parent)
    yield from _stringify_ast(node, parent_stack)
    parent_stack.pop()