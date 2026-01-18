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
class NameCollector(NodeVisitor):

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Import(self, node: Import) -> None:
        for name in node.names:
            self.names.add(name.asname or name.name)

    def visit_ImportFrom(self, node: ImportFrom) -> None:
        for name in node.names:
            self.names.add(name.asname or name.name)

    def visit_Assign(self, node: Assign) -> None:
        for target in node.targets:
            if isinstance(target, Name):
                self.names.add(target.id)

    def visit_NamedExpr(self, node: NamedExpr) -> Any:
        if isinstance(node.target, Name):
            self.names.add(node.target.id)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ClassDef) -> None:
        pass