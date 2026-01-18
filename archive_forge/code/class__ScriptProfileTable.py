import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
class _ScriptProfileTable:

    def __init__(self, cols: List[_ScriptProfileColumn], source_range: List[int]):
        self.cols = cols
        self.source_range = source_range

    def dump_string(self):
        outputs: List[str] = []
        cells: List[Tuple[str, Dict[int, str]]] = []
        header_buffer = ''
        for col in self.cols:
            header, rows = col.materialize()
            header_buffer += header
            cells.append((header, dict(rows)))
        outputs.append(header_buffer)
        outputs.append(pad('', len(header_buffer), 0, '='))
        for line in self.source_range:
            row_buffer = ''
            for header, rows in cells:
                cell = rows.get(line)
                if cell is None:
                    row_buffer += pad('', len(header))
                else:
                    row_buffer += cell
            outputs.append(row_buffer)
        return '\n'.join(outputs)