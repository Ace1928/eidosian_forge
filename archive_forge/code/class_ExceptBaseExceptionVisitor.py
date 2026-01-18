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
class ExceptBaseExceptionVisitor(ast.NodeVisitor):

    def __init__(self, except_node: ast.ExceptHandler) -> None:
        super().__init__()
        self.root = except_node
        self._re_raised = False

    def re_raised(self) -> bool:
        self.visit(self.root)
        return self._re_raised

    def visit_Raise(self, node: ast.Raise):
        """If we find a corresponding `raise` or `raise e` where e was from
        `except BaseException as e:` then we mark re_raised as True and can
        stop scanning."""
        if node.exc is None or (isinstance(node.exc, ast.Name) and node.exc.id == self.root.name):
            self._re_raised = True
            return
        return super().generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node is not self.root:
            return
        return super().generic_visit(node)