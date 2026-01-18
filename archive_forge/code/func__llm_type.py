import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
@property
def _llm_type(self) -> str:
    """Return type of llm."""
    return 'clarifai'