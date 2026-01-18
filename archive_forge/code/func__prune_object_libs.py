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
def _prune_object_libs(parent_lib: dict[str, Any], identifiers: set[str]) -> None:
    """Prune non-existing objects and empty libs from a lib's
    public.objectLibs.

    Empty object libs are pruned, but object identifiers stay.
    """
    if OBJECT_LIBS_KEY not in parent_lib:
        return
    object_libs = parent_lib[OBJECT_LIBS_KEY]
    parent_lib[OBJECT_LIBS_KEY] = {k: v for k, v in object_libs.items() if k in identifiers and v}