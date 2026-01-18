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
def _gobble_parens(self, first_token, last_token, include_all=False):
    while first_token.index > 0:
        prev = self._code.prev_token(first_token)
        next = self._code.next_token(last_token)
        if util.match_token(prev, token.OP, '(') and util.match_token(next, token.OP, ')'):
            first_token, last_token = (prev, next)
            if include_all:
                continue
        break
    return (first_token, last_token)