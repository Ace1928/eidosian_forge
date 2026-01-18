from typing import Any, Callable, Dict, Type
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.language_models.llms import BaseLLM
def _import_yuan2() -> Type[BaseLLM]:
    from langchain_community.llms.yuan2 import Yuan2
    return Yuan2