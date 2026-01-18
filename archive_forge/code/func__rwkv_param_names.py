from typing import Any, Dict, List, Mapping, Optional, Set
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
@staticmethod
def _rwkv_param_names() -> Set[str]:
    """Get the identifying parameters."""
    return {'verbose'}