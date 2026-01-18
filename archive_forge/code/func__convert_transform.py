from __future__ import annotations
import collections.abc
import uuid
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from functools import lru_cache
from typing import (
import attrs
from attrs import define, field
from fontTools.misc.arrayTools import unionRect
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen, ControlBoundsPen
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import OBJECT_LIBS_KEY
from ufoLib2.typing import Drawable, GlyphSet, HasIdentifier
def _convert_transform(t: Transform | Sequence[float]) -> Transform:
    """Return a passed-in Transform as is, otherwise convert a sequence of
    numbers to a Transform if need be."""
    return t if isinstance(t, Transform) else Transform(*t)