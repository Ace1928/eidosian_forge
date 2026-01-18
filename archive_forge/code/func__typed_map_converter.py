from __future__ import annotations
from typing import Any, Callable, Mapping
import numbers
from attrs import evolve, field, frozen
from rpds import HashTrieMap
from jsonschema.exceptions import UndefinedTypeCheck
def _typed_map_converter(init_val: Mapping[str, Callable[[TypeChecker, Any], bool]]) -> HashTrieMap[str, Callable[[TypeChecker, Any], bool]]:
    return HashTrieMap.convert(init_val)