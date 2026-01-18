from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
class FunctionNonLocals(ast.NodeVisitor):
    """Get the nonlocal variables accessed of a function."""

    def __init__(self) -> None:
        self.nonlocals: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)