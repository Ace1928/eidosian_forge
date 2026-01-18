import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
def _handle_getitem_users(self, node: torch.fx.Node) -> List[TensorArgument]:
    meta_val = node.meta['val']
    idx_to_name = {}
    for user in node.users:
        assert user.target is operator.getitem, f'User node {user} of {node} is incorrect'
        idx_to_name[user.args[1]] = user.name
    for idx, _ in enumerate(meta_val):
        if idx not in idx_to_name:
            idx_to_name[idx] = f'{node.name}_unused_{idx}'
    arg_list = []
    for i, element_meta_val in enumerate(meta_val):
        arg_list.append(self.serialize_tensor_output(idx_to_name[i], element_meta_val))
    return arg_list