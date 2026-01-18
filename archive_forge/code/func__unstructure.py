from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def _unstructure(self, converter: Converter) -> dict[str, Any]:
    test = converter.unstructure(b'\x00')
    if isinstance(test, bytes):
        return dict(self)
    elif not isinstance(test, str):
        raise NotImplementedError(type(test))
    data: dict[str, Any] = _unstructure_data(self, converter)
    return data