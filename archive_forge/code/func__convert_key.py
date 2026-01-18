from __future__ import annotations
import dataclasses
import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from torch._logging import LazyString
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import sarif
@_beartype.beartype
def _convert_key(object: Union[Dict[str, Any], Any], convert: Callable[[str], str]) -> Union[Dict[str, Any], Any]:
    """Convert and update keys in a dictionary with "convert".

    Any value that is a dictionary will be recursively updated.
    Any value that is a list will be recursively searched.

    Args:
        object: The object to update.
        convert: The function to convert the keys, e.g. `kebab_case_to_snake_case`.

    Returns:
        The updated object.
    """
    if not isinstance(object, Dict):
        return object
    new_dict = {}
    for k, v in object.items():
        new_k = convert(k)
        if isinstance(v, Dict):
            new_v = _convert_key(v, convert)
        elif isinstance(v, List):
            new_v = [_convert_key(elem, convert) for elem in v]
        else:
            new_v = v
        if new_v is None:
            continue
        if new_v == -1:
            continue
        new_dict[new_k] = new_v
    return new_dict