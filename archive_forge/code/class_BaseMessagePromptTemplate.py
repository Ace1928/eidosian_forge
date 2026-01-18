from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from langchain_core._api import deprecated
from langchain_core.load import Serializable
from langchain_core.messages import (
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import StringPromptTemplate, get_template_variables
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env
class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variables.
        """

    def pretty_repr(self, html: bool=False) -> str:
        """Human-readable representation."""
        raise NotImplementedError

    def pretty_print(self) -> None:
        print(self.pretty_repr(html=is_interactive_env()))

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other