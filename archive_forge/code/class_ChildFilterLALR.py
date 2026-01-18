from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
class ChildFilterLALR(ChildFilter):
    """Optimized childfilter for LALR (assumes no duplication in parse tree, so it's safe to change it)"""

    def __call__(self, children):
        filtered = []
        for i, to_expand, add_none in self.to_include:
            if add_none:
                filtered += [None] * add_none
            if to_expand:
                if filtered:
                    filtered += children[i].children
                else:
                    filtered = children[i].children
            else:
                filtered.append(children[i])
        if self.append_none:
            filtered += [None] * self.append_none
        return self.node_builder(filtered)