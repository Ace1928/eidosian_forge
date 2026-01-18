from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def _pp_get_meta(self, children):
    for c in children:
        if self.node_filter is not None and (not self.node_filter(c)):
            continue
        if isinstance(c, Tree):
            if not c.meta.empty:
                return c.meta
        elif isinstance(c, Token):
            return c
        elif hasattr(c, '__lark_meta__'):
            return c.__lark_meta__()