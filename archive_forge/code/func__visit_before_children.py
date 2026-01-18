import ast
import numbers
import sys
import token
from ast import Module
from typing import Callable, List, Union, cast, Optional, Tuple, TYPE_CHECKING
import six
from . import util
from .asttokens import ASTTokens
from .util import AstConstant
from .astroid_compat import astroid_node_classes as nc, BaseContainer as AstroidBaseContainer
def _visit_before_children(self, node, parent_token):
    col = getattr(node, 'col_offset', None)
    token = self._code.get_token_from_utf8(node.lineno, col) if col is not None else None
    if not token and util.is_module(node):
        token = self._code.get_token(1, 0)
    return (token or parent_token, token)