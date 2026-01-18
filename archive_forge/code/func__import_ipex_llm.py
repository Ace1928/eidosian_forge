from typing import Any, Callable, Dict, Type
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM
def _import_ipex_llm() -> Type[BaseLLM]:
    from langchain_community.llms.ipex_llm import IpexLLM
    return IpexLLM