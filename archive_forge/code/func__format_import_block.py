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
def _format_import_block(globals: Dict[str, Any], importer: Importer):
    import_strs: Set[str] = set()
    for name, obj in globals.items():
        import_strs.add(_format_import_statement(name, obj, importer))
    return '\n'.join(sorted(import_strs))