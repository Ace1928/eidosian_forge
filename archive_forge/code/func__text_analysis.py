from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def _text_analysis(self, text: str) -> Dict:
    poller = self.text_analytics_client.begin_analyze_healthcare_entities([{'id': '1', 'language': 'en', 'text': text}])
    result = poller.result()
    res_dict = {}
    docs = [doc for doc in result if not doc.is_error]
    if docs is not None:
        res_dict['entities'] = [f'{x.text} is a healthcare entity of type {x.category}' for y in docs for x in y.entities]
    return res_dict