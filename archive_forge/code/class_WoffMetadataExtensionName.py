from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
@define
class WoffMetadataExtensionName(AttrDictMixin):
    text: str
    language: Optional[str] = None
    dir: Optional[str] = None
    class_: Optional[str] = field(default=None, metadata={'rename_attr': 'class'})