import warnings
from typing import Any, Callable, Dict, Type
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.llms import BaseLLM
from langchain.utils.interactive_env import is_interactive_env
def _import_huggingface_hub() -> Any:
    from langchain_community.llms.huggingface_hub import HuggingFaceHub
    return HuggingFaceHub