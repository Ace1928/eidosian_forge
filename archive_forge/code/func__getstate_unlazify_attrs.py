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
def _getstate_unlazify_attrs(self: Any) -> Dict[str, Any]:
    if self._lazy:
        self.unlazify()
    return {a.name: getattr(self, a.name) if a.init else a.default for a in attrs.fields(self.__class__)}