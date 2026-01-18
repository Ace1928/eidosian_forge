import base64
from typing import Dict, Optional
from urllib.parse import quote
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def _filter_results(self, res: dict) -> list:
    output = []
    types = self.json_result_types if self.json_result_types is not None else []
    for task in res.get('tasks', []):
        for result in task.get('result', []):
            for item in result.get('items', []):
                if len(types) == 0 or item.get('type', '') in types:
                    self._cleanup_unnecessary_items(item)
                    if len(item) != 0:
                        output.append(item)
                if self.top_count is not None and len(output) >= self.top_count:
                    break
    return output