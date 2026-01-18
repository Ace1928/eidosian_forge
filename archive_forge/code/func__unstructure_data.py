from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def _unstructure_data(value: Any, converter: Converter) -> Any:
    if isinstance(value, bytes):
        return {'type': DATA_LIB_KEY, 'data': converter.unstructure(value)}
    elif isinstance(value, (list, tuple)):
        return [_unstructure_data(v, converter) for v in value]
    elif isinstance(value, Mapping):
        return {k: _unstructure_data(v, converter) for k, v in value.items()}
    return value