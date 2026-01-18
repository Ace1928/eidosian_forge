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
def _new_doc() -> Document:
    from ..io import curdoc
    doc = Document()
    callbacks = curdoc().callbacks._js_event_callbacks
    doc.callbacks._js_event_callbacks.update(callbacks)
    return doc