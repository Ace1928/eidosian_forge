from __future__ import annotations
import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch
from ._config import global_config
from ._transformer import TypeguardTransformer
class TypeguardLoader(SourceFileLoader):

    @staticmethod
    def source_to_code(data: Buffer | str | ast.Module | ast.Expression | ast.Interactive, path: Buffer | str | PathLike[str]='<string>') -> CodeType:
        if isinstance(data, (ast.Module, ast.Expression, ast.Interactive)):
            tree = data
        else:
            if isinstance(data, str):
                source = data
            else:
                source = decode_source(data)
            tree = _call_with_frames_removed(ast.parse, source, path, 'exec')
        tree = TypeguardTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        if global_config.debug_instrumentation and sys.version_info >= (3, 9):
            print(f'Source code of {path!r} after instrumentation:\n----------------------------------------------', file=sys.stderr)
            print(ast.unparse(tree), file=sys.stderr)
            print('----------------------------------------------', file=sys.stderr)
        return _call_with_frames_removed(compile, tree, path, 'exec', 0, dont_inherit=True)

    def exec_module(self, module: ModuleType) -> None:
        with patch('importlib._bootstrap_external.cache_from_source', optimized_cache_from_source):
            super().exec_module(module)