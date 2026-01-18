from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
class GeneratorDetector(NodeVisitor):
    """Detects if a function node is a generator function."""
    contains_yields: bool = False
    in_root_function: bool = False

    def visit_Yield(self, node: Yield) -> Any:
        self.contains_yields = True

    def visit_YieldFrom(self, node: YieldFrom) -> Any:
        self.contains_yields = True

    def visit_ClassDef(self, node: ClassDef) -> Any:
        pass

    def visit_FunctionDef(self, node: FunctionDef | AsyncFunctionDef) -> Any:
        if not self.in_root_function:
            self.in_root_function = True
            self.generic_visit(node)
            self.in_root_function = False

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)