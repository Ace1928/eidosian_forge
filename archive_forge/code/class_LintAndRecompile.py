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
class LintAndRecompile(ReversibleTransformation):
    """
    Transformation that does nothing except linting and recompiling the graph module.
    """
    preserves_computation = True

    def transform(self, graph_module: 'GraphModule') -> 'GraphModule':
        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module

    def reverse(self, graph_module: 'GraphModule') -> 'GraphModule':
        return self.transform(graph_module)