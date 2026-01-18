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
class FromCurdoc:
    """ This class merely provides a non-None default value for ``theme``
    arguments, since ``None`` itself is a meaningful value for users to pass.

    """
    pass