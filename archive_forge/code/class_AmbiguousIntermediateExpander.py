from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
class AmbiguousIntermediateExpander:
    """
    Propagate ambiguous intermediate nodes and their derivations up to the
    current rule.

    In general, converts

    rule
      _iambig
        _inter
          someChildren1
          ...
        _inter
          someChildren2
          ...
      someChildren3
      ...

    to

    _ambig
      rule
        someChildren1
        ...
        someChildren3
        ...
      rule
        someChildren2
        ...
        someChildren3
        ...
      rule
        childrenFromNestedIambigs
        ...
        someChildren3
        ...
      ...

    propagating up any nested '_iambig' nodes along the way.
    """

    def __init__(self, tree_class, node_builder):
        self.node_builder = node_builder
        self.tree_class = tree_class

    def __call__(self, children):

        def _is_iambig_tree(child):
            return hasattr(child, 'data') and child.data == '_iambig'

        def _collapse_iambig(children):
            """
            Recursively flatten the derivations of the parent of an '_iambig'
            node. Returns a list of '_inter' nodes guaranteed not
            to contain any nested '_iambig' nodes, or None if children does
            not contain an '_iambig' node.
            """
            if children and _is_iambig_tree(children[0]):
                iambig_node = children[0]
                result = []
                for grandchild in iambig_node.children:
                    collapsed = _collapse_iambig(grandchild.children)
                    if collapsed:
                        for child in collapsed:
                            child.children += children[1:]
                        result += collapsed
                    else:
                        new_tree = self.tree_class('_inter', grandchild.children + children[1:])
                        result.append(new_tree)
                return result
        collapsed = _collapse_iambig(children)
        if collapsed:
            processed_nodes = [self.node_builder(c.children) for c in collapsed]
            return self.tree_class('_ambig', processed_nodes)
        return self.node_builder(children)