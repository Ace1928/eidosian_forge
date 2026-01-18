import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def _create_llm_result(self, response: List[dict]) -> LLMResult:
    """Create the LLMResult from the choices and prompts."""
    generations = []
    for res in response:
        results = res.get('results')
        if results:
            finish_reason = results[0].get('stop_reason')
            gen = Generation(text=results[0].get('generated_text'), generation_info={'finish_reason': finish_reason})
            generations.append([gen])
    final_token_usage = self._extract_token_usage(response)
    llm_output = {'token_usage': final_token_usage, 'model_id': self.model_id, 'deployment_id': self.deployment_id}
    return LLMResult(generations=generations, llm_output=llm_output)