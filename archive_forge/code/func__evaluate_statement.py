from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def _evaluate_statement(self, node: mparser.BaseNode) -> T.Union[str, bool, int, T.List[str]]:
    if isinstance(node, mparser.BaseStringNode):
        return node.value
    elif isinstance(node, mparser.BooleanNode):
        return node.value
    elif isinstance(node, mparser.NumberNode):
        return node.value
    elif isinstance(node, mparser.ParenthesizedNode):
        return self._evaluate_statement(node.inner)
    elif isinstance(node, mparser.ArrayNode):
        return [self._evaluate_statement(arg) for arg in node.args.arguments]
    elif isinstance(node, mparser.IdNode):
        return self.scope[node.value]
    elif isinstance(node, mparser.ArithmeticNode):
        l = self._evaluate_statement(node.left)
        r = self._evaluate_statement(node.right)
        if node.operation == 'add':
            if isinstance(l, str) and isinstance(r, str) or (isinstance(l, list) and isinstance(r, list)):
                return l + r
        elif node.operation == 'div':
            if isinstance(l, str) and isinstance(r, str):
                return os.path.join(l, r)
    raise EnvironmentException('Unsupported node type')