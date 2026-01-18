from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
def _reset_keeping_doc(self) -> None:
    """ Reset output modes but DO NOT replace the default Document

        """
    self._file = None
    self._notebook = False
    self._notebook_type = None