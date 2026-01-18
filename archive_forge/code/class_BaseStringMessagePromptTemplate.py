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
class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    """Base class for message prompt templates that use a string prompt template."""
    prompt: StringPromptTemplate
    'String prompt template.'
    additional_kwargs: dict = Field(default_factory=dict)
    'Additional keyword arguments to pass to the prompt template.'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']

    @classmethod
    def from_template(cls: Type[MessagePromptTemplateT], template: str, template_format: str='f-string', partial_variables: Optional[Dict[str, Any]]=None, **kwargs: Any) -> MessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
            partial_variables: A dictionary of variables that can be used to partially
                               fill in the template. For example, if the template is
                              `"{variable1} {variable2}"`, and `partial_variables` is
                              `{"variable1": "foo"}`, then the final prompt will be
                              `"foo {variable2}"`.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_template(template, template_format=template_format, partial_variables=partial_variables)
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(cls: Type[MessagePromptTemplateT], template_file: Union[str, Path], input_variables: List[str], **kwargs: Any) -> MessagePromptTemplateT:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt = PromptTemplate.from_file(template_file, input_variables)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        """
        Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return self.prompt.input_variables

    def pretty_repr(self, html: bool=False) -> str:
        title = self.__class__.__name__.replace('MessagePromptTemplate', ' Message')
        title = get_msg_title_repr(title, bold=html)
        return f'{title}\n\n{self.prompt.pretty_repr(html=html)}'