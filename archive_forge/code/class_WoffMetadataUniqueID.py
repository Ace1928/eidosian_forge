from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
@define
class WoffMetadataUniqueID(AttrDictMixin):
    id: str