from __future__ import annotations
import inspect
import uuid
from typing import (
from typing_extensions import TypedDict
from langchain_core._api import deprecated
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.json_schema import dereference_refs
def _rm_titles(kv: dict, prev_key: str='') -> dict:
    new_kv = {}
    for k, v in kv.items():
        if k == 'title':
            if isinstance(v, dict) and prev_key == 'properties' and ('title' in v.keys()):
                new_kv[k] = _rm_titles(v, k)
            else:
                continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv