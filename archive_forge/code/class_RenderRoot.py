from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
@dataclass
class RenderRoot:
    """ Encapsulate data needed for embedding a Bokeh document root.

    Values for ``name`` or ``tags`` are optional. They may be useful for
    querying a collection of roots to find a specific one to embed.

    """
    elementid: ID
    id: ID = field(compare=False)
    name: str | None = field(default='', compare=False)
    tags: list[Any] = field(default_factory=list, compare=False)

    def __post_init__(self):
        self.name = self.name or ''