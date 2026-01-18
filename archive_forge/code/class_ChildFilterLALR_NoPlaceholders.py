from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
class ChildFilterLALR_NoPlaceholders(ChildFilter):
    """Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)"""

    def __init__(self, to_include, node_builder):
        self.node_builder = node_builder
        self.to_include = to_include

    def __call__(self, children):
        filtered = []
        for i, to_expand in self.to_include:
            if to_expand:
                if filtered:
                    filtered += children[i].children
                else:
                    filtered = children[i].children
            else:
                filtered.append(children[i])
        return self.node_builder(filtered)