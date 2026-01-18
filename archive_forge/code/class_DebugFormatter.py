import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
class DebugFormatter:

    def __init__(self, handler):
        self.fopen = handler.fopen
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        with self.fopen('fx_graph_runnable.py') as fd:
            save_graph_repro(fd, gm, inputs, 'inductor')
        with self.fopen('fx_graph_readable.py') as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        with self.fopen('fx_graph_transformed.py') as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList):
        self._write_ir('ir_pre_fusion.txt', nodes)

    def ir_post_fusion(self, nodes: SchedulerNodeList):
        self._write_ir('ir_post_fusion.txt', nodes)

    def _write_ir(self, filename: str, nodes: SchedulerNodeList):
        with self.fopen(filename) as fd:
            log.info('Writing debug ir to  %s', fd.name)
            for node in nodes:
                fd.write(node.debug_str())
                fd.write('\n\n\n')

    def graph_diagram(self, nodes: SchedulerNodeList):
        draw_buffers(nodes, fname=self.filename('graph_diagram.svg'))

    def draw_orig_fx_graph(self, gm: torch.fx.GraphModule, nodes: SchedulerNodeList):
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(gm, fname=self.filename('orig_fx_graph_diagram.svg'), clear_meta=False, prog=GRAPHVIZ_COMMAND_SCALABLE, parse_stack_trace=True, dot_graph_shape=config.trace.dot_graph_shape)

    def output_code(self, filename):
        shutil.copy(filename, self.filename('output_code.py'))