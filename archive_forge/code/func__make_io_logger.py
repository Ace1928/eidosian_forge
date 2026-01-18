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
def _make_io_logger(self, name: str) -> Callable[..., None]:

    def logger(*args: Any, **kwargs: Any) -> None:
        log.info(self._logger_text, self._runner.path, name)
    return logger