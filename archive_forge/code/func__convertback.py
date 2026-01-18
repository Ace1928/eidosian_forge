import json
from array import array
from enum import Enum, auto
from typing import Any
def _convertback(data_type: DataType, inx: Any) -> Any:
    if data_type == DataType.FLOAT:
        return inx[0]
    if data_type == DataType.LIST:
        return inx.tolist()
    if data_type == DataType.TUPLE:
        return tuple(inx)
    return inx