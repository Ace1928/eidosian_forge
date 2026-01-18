import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
class DeepCopy(ReversibleTransformation):
    """
    Transformation that does nothing except making a deepcopy of the graph module.
    """
    preserves_computation = True

    def transform(self, graph_module: 'GraphModule') -> 'GraphModule':
        clone = copy.deepcopy(graph_module)
        for n1, n2 in zip(graph_module.graph.nodes, clone.graph.nodes):
            if hasattr(n1, 'transformations'):
                n2.transformations = n1.transformations
        return clone

    def reverse(self, graph_module: 'GraphModule') -> 'GraphModule':
        return self.transform(graph_module)