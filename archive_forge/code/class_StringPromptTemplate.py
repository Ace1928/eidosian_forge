from __future__ import annotations
import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable, Dict, List, Set, Tuple, Type
import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env
class StringPromptTemplate(BasePromptTemplate, ABC):
    """String prompt that exposes the format method, returning a prompt."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'base']

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=await self.aformat(**kwargs))

    def pretty_repr(self, html: bool=False) -> str:
        dummy_vars = {input_var: '{' + f'{input_var}' + '}' for input_var in self.input_variables}
        if html:
            dummy_vars = {k: get_colored_text(v, 'yellow') for k, v in dummy_vars.items()}
        return self.format(**dummy_vars)

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))