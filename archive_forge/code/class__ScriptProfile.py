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
class _ScriptProfile:

    def __init__(self):
        self.profile = classes.profiling._ScriptProfile()

    def enable(self):
        self.profile.enable()

    def disable(self):
        self.profile.disable()

    def dump_string(self) -> str:
        outputs: List[str] = []
        for source_stats in self.profile._dump_stats():
            source_ref = source_stats.source()
            source_lines = source_ref.text().splitlines()
            dedent = min([len(line) - len(line.lstrip(' ')) for line in source_lines])
            source_lines = [line[dedent:] for line in source_lines]
            start_line = source_ref.starting_lineno()
            end_line = start_line + len(source_lines)
            source_range = range(start_line, end_line)
            lineno = _ScriptProfileColumn('Line #')
            hits = _ScriptProfileColumn('Hits')
            time_ns = _ScriptProfileColumn('Time (ns)')
            line_contents = _ScriptProfileColumn('Line Contents', 0, 1)
            stats = source_stats.line_map()
            for line in source_range:
                lineno.add_row(line, line)
                line_contents.add_row(line, source_lines[line - start_line])
                stat = stats.get(line)
                if stat is not None:
                    hits.add_row(line, stat.count())
                    time_ns.add_row(line, stat.duration_ns())
            table = _ScriptProfileTable([lineno, hits, time_ns, line_contents], list(source_range))
            outputs.append(table.dump_string())
        return '\n\n'.join(outputs)

    def dump(self):
        print(self.dump_string())