from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
@attr.s
class NameFinder(ast.NodeVisitor):
    """Finds a name within a tree of nodes.

    After `.visit(node)` is called, `found` is a dict with all name nodes inside,
    key is name string, value is the node (useful for location purposes).
    """
    names: Dict[str, List[ast.Name]] = attr.ib(default=attr.Factory(dict))

    def visit_Name(self, node: ast.Name):
        self.names.setdefault(node.id, []).append(node)

    def visit(self, node):
        """Like super-visit but supports iteration over lists."""
        if not isinstance(node, list):
            return super().visit(node)
        for elem in node:
            super().visit(elem)
        return node