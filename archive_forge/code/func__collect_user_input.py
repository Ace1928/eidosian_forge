from typing import Any, Callable, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
def _collect_user_input(separator: Optional[str]=None, stop: Optional[List[str]]=None) -> str:
    """Collects and returns user input as a single string."""
    separator = separator or '\n'
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
        if stop and any((seq in line for seq in stop)):
            break
    multi_line_input = separator.join(lines)
    return multi_line_input