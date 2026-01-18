import logging
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.tools import BaseTool
from langchain_community.chat_models.openai import _import_tiktoken
from langchain_community.tools.powerbi.prompt import (
from langchain_community.utilities.powerbi import PowerBIDataset, json_to_md
def _check_cache(self, tool_input: str) -> Optional[str]:
    """Check if the input is present in the cache.

        If the value is a bad request, overwrite with the escalated version,
        if not present return None."""
    if tool_input not in self.session_cache:
        return None
    return self.session_cache[tool_input]