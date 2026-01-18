from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
@define
class WoffMetadataLicense(AttrDictMixin):
    url: Optional[str] = None
    id: Optional[str] = None
    text: List[WoffMetadataText] = field(factory=list, converter=_convert_list_of_woff_metadata_texts)