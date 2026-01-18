import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
class ChoiceCaller:
    """
    Represents a possible choice used in autotune_process.py.
    During autotuning, self.benchmark() is first called to get benchmark result,
    and if this choice is selected, self.output_node() is called to get the output_node.

    Children classes: TritonTemplateCaller, CUDATemplateCaller.
    """

    def __init__(self, name, input_nodes, layout):
        super().__init__()
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes

    def benchmark(self, *args, out) -> float:
        algo = self.to_callable()
        return do_bench(lambda: algo(*args, out=out))

    def call_name(self) -> str:
        raise NotImplementedError()

    def to_callable(self):
        raise NotImplementedError()

    def hash_key(self) -> str:
        raise NotImplementedError()

    def output_node(self) -> 'TensorBox':
        raise NotImplementedError()