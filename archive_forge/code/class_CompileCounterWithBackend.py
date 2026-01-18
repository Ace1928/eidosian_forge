import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
class CompileCounterWithBackend:

    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        from .backends.registry import lookup_backend
        self.frame_count += 1
        for node in gm.graph.nodes:
            if 'call' in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)