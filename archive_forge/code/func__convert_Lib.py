from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def _convert_Lib(value: Mapping[str, Any]) -> Lib:
    return value if isinstance(value, Lib) else Lib(value)