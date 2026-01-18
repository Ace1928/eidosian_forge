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
@contextmanager
def _use_memo(self, node: ClassDef | FunctionDef | AsyncFunctionDef) -> Generator[None, Any, None]:
    new_memo = TransformMemo(node, self._memo, self._memo.path + (node.name,))
    old_memo = self._memo
    self._memo = new_memo
    if isinstance(node, (FunctionDef, AsyncFunctionDef)):
        new_memo.should_instrument = self._target_path is None or new_memo.path == self._target_path
        if new_memo.should_instrument:
            detector = GeneratorDetector()
            detector.visit(node)
            return_annotation = deepcopy(node.returns)
            if detector.contains_yields and new_memo.name_matches(return_annotation, *generator_names):
                if isinstance(return_annotation, Subscript):
                    annotation_slice = return_annotation.slice
                    if isinstance(annotation_slice, Index):
                        annotation_slice = annotation_slice.value
                    if isinstance(annotation_slice, Tuple):
                        items = annotation_slice.elts
                    else:
                        items = [annotation_slice]
                    if len(items) > 0:
                        new_memo.yield_annotation = self._convert_annotation(items[0])
                    if len(items) > 1:
                        new_memo.send_annotation = self._convert_annotation(items[1])
                    if len(items) > 2:
                        new_memo.return_annotation = self._convert_annotation(items[2])
            else:
                new_memo.return_annotation = self._convert_annotation(return_annotation)
    if isinstance(node, AsyncFunctionDef):
        new_memo.is_async = True
    yield
    self._memo = old_memo