import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
@dataclass(frozen=True)
class ComputeRedispatchFunction:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
        result = ''
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join(['dispatchKeySet'] + [a.expr for a in exprs])
            result += f'\n// aten::{f.func}\ninline {sig.decl(is_redispatching_fn=True)} {{\n    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});\n}}\n'
        return result