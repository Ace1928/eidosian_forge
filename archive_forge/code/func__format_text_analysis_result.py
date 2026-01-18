from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _format_text_analysis_result(self, text_analysis_result: Dict) -> str:
    formatted_result = []
    if 'entities' in text_analysis_result:
        formatted_result.append(f'The text contains the following healthcare entities: {', '.join(text_analysis_result['entities'])}'.replace('\n', ' '))
    return '\n'.join(formatted_result)