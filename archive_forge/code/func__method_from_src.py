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
def _method_from_src(method_name: str, src: str, globals: Dict[str, Any], co_fields=None) -> Callable:
    globals_copy = globals.copy()
    _exec_with_source(src, globals_copy, co_fields)
    fn = globals_copy[method_name]
    del globals_copy[method_name]
    return fn