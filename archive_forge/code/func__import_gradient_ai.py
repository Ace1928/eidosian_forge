import warnings
from typing import Any, Callable, Dict, Type
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.llms import BaseLLM
from langchain.utils.interactive_env import is_interactive_env
def _import_gradient_ai() -> Any:
    from langchain_community.llms.gradient_ai import GradientLLM
    return GradientLLM