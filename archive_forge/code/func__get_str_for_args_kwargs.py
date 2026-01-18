import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _get_str_for_args_kwargs(arg):
    if isinstance(arg, tuple):
        prefix, suffix = ('|args=(\\l', ',\\n)\\l')
        arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
    elif isinstance(arg, dict):
        prefix, suffix = ('|kwargs={\\l', ',\\n}\\l')
        arg_strs_list = [f'{k}: {_format_arg(v, max_list_len=8)}' for k, v in arg.items()]
    else:
        return ''
    if skip_node_names_in_args:
        arg_strs_list = [a for a in arg_strs_list if '%' not in a]
    if len(arg_strs_list) == 0:
        return ''
    arg_strs = prefix + ',\\n'.join(arg_strs_list) + suffix
    if len(arg_strs_list) == 1:
        arg_strs = arg_strs.replace('\\l', '').replace('\\n', '')
    return arg_strs.replace('{', '\\{').replace('}', '\\}')