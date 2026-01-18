import base64
from typing import Dict, Optional
from urllib.parse import quote
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _cleanup_unnecessary_items(self, d: dict) -> dict:
    fields = self.json_result_fields if self.json_result_fields is not None else []
    if len(fields) > 0:
        for k, v in list(d.items()):
            if isinstance(v, dict):
                self._cleanup_unnecessary_items(v)
                if len(v) == 0:
                    del d[k]
            elif k not in fields:
                del d[k]
    if 'xpath' in d:
        del d['xpath']
    if 'position' in d:
        del d['position']
    if 'rectangle' in d:
        del d['rectangle']
    for k, v in list(d.items()):
        if isinstance(v, dict):
            self._cleanup_unnecessary_items(v)
    return d