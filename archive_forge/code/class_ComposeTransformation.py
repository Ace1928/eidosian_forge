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
class ComposeTransformation(ReversibleTransformation):
    preserves_computation = composition_preserves_computation
    _composition = functools.reduce(make_reduce_fn(False), transformations)
    _reverse_composition = functools.reduce(make_reduce_fn(True), reversed(transformations))

    def transform(self, graph_module):
        return ComposeTransformation._composition(graph_module)

    def reverse(self, graph_module):
        return ComposeTransformation._reverse_composition(graph_module)