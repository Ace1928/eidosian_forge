from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
def _convert_list_of_woff_metadata_credits(value: list[WoffMetadataCredit | Mapping[str, Any]]) -> list[WoffMetadataCredit]:
    return _convert_list_of_woff_metadata(WoffMetadataCredit, value)