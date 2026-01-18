import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _insert_doc_at_pattern(obj, *, message: str, pattern: str, insert_after: bool=True, directive: Optional[str]=None, skip_matches: int=0) -> str:
    if '\n' in message:
        raise ValueError(f"message shouldn't contain any newlines, since this function will insert its own linebreaks when text wrapping: {message}")
    doc = obj.__doc__.strip()
    if not doc:
        doc = ''
    if pattern == '' and insert_after:
        head = doc
        tail = ''
    else:
        tail = doc
        i = tail.find(pattern)
        skip_matches_left = skip_matches
        while i != -1:
            if insert_after:
                offset = i + len(pattern)
            else:
                offset = tail[:i].rfind('\n') + 1
            head = tail[:offset]
            tail = tail[offset:]
            skip_matches_left -= 1
            if skip_matches_left <= 0:
                break
            elif not insert_after:
                tail = tail[i - offset + len(pattern):]
            i = tail.find(pattern)
        else:
            raise ValueError(f'Pattern {pattern} not found after {skip_matches} skips in docstring {doc}')
    after_lines = list(filter(bool, tail.splitlines()))
    if len(after_lines) > 0:
        lines = after_lines
    else:
        lines = list(filter(bool, reversed(head.splitlines())))
    assert len(lines) > 0
    indent = ' ' * (len(lines[0]) - len(lines[0].lstrip()))
    message = message.strip('\n')
    if directive is not None:
        base = f'{indent}.. {directive}::\n'
        message = message.replace('\n', '\n' + indent + ' ' * 4)
        message = base + indent + ' ' * 4 + message
    else:
        message = indent + message.replace('\n', '\n' + indent)
    if insert_after ^ (pattern == '\n\n'):
        message = '\n\n' + message
    if (not insert_after) ^ (pattern == '\n\n'):
        message = message + '\n\n'
    parts = [head, message, tail]
    obj.__doc__ = ''.join(parts)