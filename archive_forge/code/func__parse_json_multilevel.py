from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _parse_json_multilevel(self, extracted_data: dict, formatted_list: list, level: int=0) -> None:
    for section, subsections in extracted_data.items():
        indentation = '  ' * level
        if isinstance(subsections, str):
            subsections = subsections.replace('\n', ',')
            formatted_list.append(f'{indentation}{section} : {subsections}')
        elif isinstance(subsections, list):
            formatted_list.append(f'{indentation}{section} : ')
            self._list_handling(subsections, formatted_list, level + 1)
        elif isinstance(subsections, dict):
            formatted_list.append(f'{indentation}{section} : ')
            self._parse_json_multilevel(subsections, formatted_list, level + 1)