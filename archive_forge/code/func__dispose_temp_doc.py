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
def _dispose_temp_doc(models: Sequence[Model]) -> None:
    for m in models:
        m._temp_document = None
        for ref in m.references():
            ref._temp_document = None