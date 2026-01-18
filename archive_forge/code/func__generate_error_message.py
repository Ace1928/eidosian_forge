import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
@staticmethod
def _generate_error_message(frame_summary: traceback.FrameSummary) -> str:
    err_lineno = frame_summary.lineno
    assert err_lineno is not None
    line = frame_summary.line
    assert line is not None
    err_line_len = len(line)
    all_src_lines = linecache.getlines(frame_summary.filename)
    tb_repr = traceback.format_exc()
    custom_msg = f"Call using an FX-traced Module, line {err_lineno} of the traced Module's generated forward function:"
    before_err = ''.join(all_src_lines[err_lineno - 2:err_lineno])
    marker = '~' * err_line_len + '~~~ <--- HERE'
    err_and_after_err = '\n'.join(all_src_lines[err_lineno:err_lineno + 2])
    return '\n'.join([tb_repr, custom_msg, before_err, marker, err_and_after_err])