import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union
import yaml
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
def _sig_decl_wrapper(sig: Union[CppSignature, ExecutorchCppSignature]) -> str:
    """
    A wrapper function to basically get `sig.decl(include_context=True)`.
    For ATen kernel, the codegen has no idea about ET contextArg, so we
    use this wrapper to add it.
    """
    if isinstance(sig, ExecutorchCppSignature):
        return sig.decl()
    returns_type = aten_cpp.returns_type(sig.func.returns).cpp_type()
    cpp_args = [a.decl() for a in sig.arguments()]
    cpp_args_str = ', '.join([contextArg.decl()] + cpp_args)
    sig_decl = f'{returns_type} {sig.name()}({cpp_args_str})'
    return sig_decl