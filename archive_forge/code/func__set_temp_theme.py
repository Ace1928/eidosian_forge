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
def _set_temp_theme(doc: Document, apply_theme: Theme | type[FromCurdoc] | None) -> None:
    _themes[doc] = doc.theme
    if apply_theme is FromCurdoc:
        from ..io import curdoc
        doc.theme = curdoc().theme
    elif isinstance(apply_theme, Theme):
        doc.theme = apply_theme