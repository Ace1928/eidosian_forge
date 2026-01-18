from __future__ import annotations
import logging # isort:skip
from contextlib import contextmanager
from os.path import basename, splitext
from types import ModuleType
from typing import Any, Callable, ClassVar
from ...core.types import PathLike
from ...document import Document
from ...io.doc import curdoc, patch_curdoc
from .code_runner import CodeRunner
from .handler import Handler
def _make_post_doc_check(self, doc: Document) -> Callable[[], None]:

    def func() -> None:
        if curdoc() is not doc:
            raise RuntimeError(f'{self._origin} at {self._runner.path!r} replaced the output document')
    return func