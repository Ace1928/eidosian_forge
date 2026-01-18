from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_cognitive_services.utils import (
def _format_image_analysis_result(self, image_analysis_result: Dict) -> str:
    formatted_result = []
    if 'caption' in image_analysis_result:
        formatted_result.append('Caption: ' + image_analysis_result['caption'])
    if 'objects' in image_analysis_result and len(image_analysis_result['objects']) > 0:
        formatted_result.append('Objects: ' + ', '.join(image_analysis_result['objects']))
    if 'tags' in image_analysis_result and len(image_analysis_result['tags']) > 0:
        formatted_result.append('Tags: ' + ', '.join(image_analysis_result['tags']))
    if 'text' in image_analysis_result and len(image_analysis_result['text']) > 0:
        formatted_result.append('Text: ' + ', '.join(image_analysis_result['text']))
    return '\n'.join(formatted_result)