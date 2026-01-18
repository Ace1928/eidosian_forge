from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_cognitive_services.utils import (
def _parse_kv_pairs(self, kv_pairs: List[Any]) -> List[Any]:
    result = []
    for kv_pair in kv_pairs:
        key = kv_pair.key.content if kv_pair.key else ''
        value = kv_pair.value.content if kv_pair.value else ''
        result.append((key, value))
    return result