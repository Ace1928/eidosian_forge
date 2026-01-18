from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_cognitive_services.utils import (
def _format_document_analysis_result(self, document_analysis_result: Dict) -> str:
    formatted_result = []
    if 'content' in document_analysis_result:
        formatted_result.append(f'Content: {document_analysis_result['content']}'.replace('\n', ' '))
    if 'tables' in document_analysis_result:
        for i, table in enumerate(document_analysis_result['tables']):
            formatted_result.append(f'Table {i}: {table}'.replace('\n', ' '))
    if 'key_value_pairs' in document_analysis_result:
        for kv_pair in document_analysis_result['key_value_pairs']:
            formatted_result.append(f'{kv_pair[0]}: {kv_pair[1]}'.replace('\n', ' '))
    return '\n'.join(formatted_result)