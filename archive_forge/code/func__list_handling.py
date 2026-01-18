from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _list_handling(self, subsection_list: list, formatted_list: list, level: int) -> None:
    for list_item in subsection_list:
        if isinstance(list_item, dict):
            self._parse_json_multilevel(list_item, formatted_list, level)
        elif isinstance(list_item, list):
            self._list_handling(list_item, formatted_list, level + 1)
        else:
            formatted_list.append(f'{'  ' * level}{list_item}')