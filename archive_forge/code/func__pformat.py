import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
def _pformat(obj: Any, indent: int, width: int, curr_indent: int=0) -> str:
    assert is_dataclass(obj), f'obj should be a dataclass, received: {type(obj)}'
    class_name = obj.__class__.__name__
    curr_indent += len(class_name) + 1
    fields_list = [(f.name, getattr(obj, f.name)) for f in fields(obj) if f.repr]
    fields_str = []
    for name, attr in fields_list:
        _curr_indent = curr_indent + len(name) + 1
        if is_dataclass(attr):
            str_repr = _pformat(attr, indent, width, _curr_indent)
        elif isinstance(attr, dict):
            str_repr = _format_dict(attr, indent, width, _curr_indent)
        elif isinstance(attr, (list, set, tuple)):
            str_repr = _format_list(attr, indent, width, _curr_indent)
        else:
            str_repr = repr(attr)
        fields_str.append(f'{name}={str_repr}')
    indent_str = curr_indent * ' '
    body = f',\n{indent_str}'.join(fields_str)
    return f'{class_name}({body})'