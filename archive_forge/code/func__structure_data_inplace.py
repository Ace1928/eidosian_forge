from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def _structure_data_inplace(key: Union[int, str], value: Any, container: Any, converter: Converter) -> None:
    if isinstance(value, list):
        for i, v in enumerate(value):
            _structure_data_inplace(i, v, value, converter)
    elif is_data_dict(value):
        container[key] = converter.structure(value['data'], bytes)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _structure_data_inplace(k, v, value, converter)