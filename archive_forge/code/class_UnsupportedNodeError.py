import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
class UnsupportedNodeError(NotSupportedError):

    def __init__(self, ctx, offending_node, reason=''):
        node_type = type(offending_node)
        range_len = len(node_start_tokens.get(node_type, ' '))
        source_range = ctx.make_range(offending_node.lineno, offending_node.col_offset, offending_node.col_offset + range_len)
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        msg = f"{feature_name} {(reason + ' ' if reason else '')}aren't supported"
        super().__init__(source_range, msg)