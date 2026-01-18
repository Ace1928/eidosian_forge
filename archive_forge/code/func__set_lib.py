from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def _set_lib(self: Any, value: Mapping[str, Any]) -> None:
    self._lib = _convert_Lib(value)