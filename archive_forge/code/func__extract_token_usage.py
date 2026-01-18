import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
@staticmethod
def _extract_token_usage(response: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
    if response is None:
        return {'generated_token_count': 0, 'input_token_count': 0}
    input_token_count = 0
    generated_token_count = 0

    def get_count_value(key: str, result: Dict[str, Any]) -> int:
        return result.get(key, 0) or 0
    for res in response:
        results = res.get('results')
        if results:
            input_token_count += get_count_value('input_token_count', results[0])
            generated_token_count += get_count_value('generated_token_count', results[0])
    return {'generated_token_count': generated_token_count, 'input_token_count': input_token_count}