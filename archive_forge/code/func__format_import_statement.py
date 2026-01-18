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
def _format_import_statement(name: str, obj: Any, importer: Importer) -> str:
    if name in _custom_builtins:
        return _custom_builtins[name].import_str
    if _is_from_torch(name):
        return 'import torch'
    module_name, attr_name = importer.get_name(obj)
    return f'from {module_name} import {attr_name} as {name}'