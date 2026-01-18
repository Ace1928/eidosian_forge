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
class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name, line):
        super().__init__(line)
        self.name = name

    def __call__(self):
        if all((self.name not in x for x in (V.graph.removed_buffers, V.kernel.removed_buffers, V.graph.inplaced_to_remove, V.kernel.inplaced_to_remove))):
            return self.line
        return None

    def _new_line(self, line):
        return DeferredLine(self.name, line)