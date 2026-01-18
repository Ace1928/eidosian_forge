import sys
import types
from typing import List
import torch
class _QEngineProp:

    def __get__(self, obj, objtype) -> str:
        return _get_qengine_str(torch._C._get_qengine())

    def __set__(self, obj, val: str) -> None:
        torch._C._set_qengine(_get_qengine_id(val))