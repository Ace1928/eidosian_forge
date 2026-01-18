from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
class PropagatePositions:

    def __init__(self, node_builder, node_filter=None):
        self.node_builder = node_builder
        self.node_filter = node_filter

    def __call__(self, children):
        res = self.node_builder(children)
        if isinstance(res, Tree):
            res_meta = res.meta
            first_meta = self._pp_get_meta(children)
            if first_meta is not None:
                if not hasattr(res_meta, 'line'):
                    res_meta.line = getattr(first_meta, 'container_line', first_meta.line)
                    res_meta.column = getattr(first_meta, 'container_column', first_meta.column)
                    res_meta.start_pos = getattr(first_meta, 'container_start_pos', first_meta.start_pos)
                    res_meta.empty = False
                res_meta.container_line = getattr(first_meta, 'container_line', first_meta.line)
                res_meta.container_column = getattr(first_meta, 'container_column', first_meta.column)
                res_meta.container_start_pos = getattr(first_meta, 'container_start_pos', first_meta.start_pos)
            last_meta = self._pp_get_meta(reversed(children))
            if last_meta is not None:
                if not hasattr(res_meta, 'end_line'):
                    res_meta.end_line = getattr(last_meta, 'container_end_line', last_meta.end_line)
                    res_meta.end_column = getattr(last_meta, 'container_end_column', last_meta.end_column)
                    res_meta.end_pos = getattr(last_meta, 'container_end_pos', last_meta.end_pos)
                    res_meta.empty = False
                res_meta.container_end_line = getattr(last_meta, 'container_end_line', last_meta.end_line)
                res_meta.container_end_column = getattr(last_meta, 'container_end_column', last_meta.end_column)
                res_meta.container_end_pos = getattr(last_meta, 'container_end_pos', last_meta.end_pos)
        return res

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