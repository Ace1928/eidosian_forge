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
class ChatPromptTemplate(BaseChatPromptTemplate):
    """Prompt template for chat models.

    Use to create flexible templated prompts for chat models.

    Examples:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ])

            prompt_value = template.invoke(
                {
                    "name": "Bob",
                    "user_input": "What is your name?"
                }
            )
            # Output:
            # ChatPromptValue(
            #    messages=[
            #        SystemMessage(content='You are a helpful AI bot. Your name is Bob.'),
            #        HumanMessage(content='Hello, how are you doing?'),
            #        AIMessage(content="I'm doing well, thanks!"),
            #        HumanMessage(content='What is your name?')
            #    ]
            #)

    Messages Placeholder:

        .. code-block:: python

            # In addition to Human/AI/Tool/Function messages,
            # you can initialize the template with a MessagesPlaceholder
            # either using the class directly or with the shorthand tuple syntax:

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot."),
                # Means the template will receive an optional list of messages under
                # the "conversation" key
                ("placeholder", "{conversation}")
                # Equivalently:
                # MessagesPlaceholder(variable_name="conversation", optional=True)
            ])

            prompt_value = template.invoke(
                {
                    "conversation": [
                        ("human", "Hi!"),
                        ("ai", "How can I assist you today?"),
                        ("human", "Can you make me an ice cream sundae?"),
                        ("ai", "No.")
                    ]
                }
            )

            # Output:
            # ChatPromptValue(
            #    messages=[
            #        SystemMessage(content='You are a helpful AI bot.'),
            #        HumanMessage(content='Hi!'),
            #        AIMessage(content='How can I assist you today?'),
            #        HumanMessage(content='Can you make me an ice cream sundae?'),
            #        AIMessage(content='No.'),
            #    ]
            #)

    Single-variable template:

        If your prompt has only a single input variable (i.e., 1 instance of "{variable_nams}"),
        and you invoke the template with a non-dict object, the prompt template will
        inject the provided argument into that variable location.


        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is Carl."),
                ("human", "{user_input}"),
            ])

            prompt_value = template.invoke("Hello, there!")
            # Equivalent to
            # prompt_value = template.invoke({"user_input": "Hello, there!"})

            # Output:
            #  ChatPromptValue(
            #     messages=[
            #         SystemMessage(content='You are a helpful AI bot. Your name is Carl.'),
            #         HumanMessage(content='Hello, there!'),
            #     ]
            # )

    """
    input_variables: List[str]
    'List of input variables in template messages. Used for validation.'
    messages: List[MessageLike]
    'List of messages consisting of either message prompt templates or messages.'
    validate_template: bool = False
    'Whether or not to try validating the template.'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'prompts', 'chat']

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        if isinstance(other, ChatPromptTemplate):
            return ChatPromptTemplate(messages=self.messages + other.messages)
        elif isinstance(other, (BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate)):
            return ChatPromptTemplate(messages=self.messages + [other])
        elif isinstance(other, (list, tuple)):
            _other = ChatPromptTemplate.from_messages(other)
            return ChatPromptTemplate(messages=self.messages + _other.messages)
        elif isinstance(other, str):
            prompt = HumanMessagePromptTemplate.from_template(other)
            return ChatPromptTemplate(messages=self.messages + [prompt])
        else:
            raise NotImplementedError(f'Unsupported operand type for +: {type(other)}')

    @root_validator(pre=True)
    def validate_input_variables(cls, values: dict) -> dict:
        """Validate input variables.

        If input_variables is not set, it will be set to the union of
        all input variables in the messages.

        Args:
            values: values to validate.

        Returns:
            Validated values.
        """
        messages = values['messages']
        input_vars = set()
        input_types: Dict[str, Any] = values.get('input_types', {})
        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = List[AnyMessage]
        if 'partial_variables' in values:
            input_vars = input_vars - set(values['partial_variables'])
        if 'input_variables' in values and values.get('validate_template'):
            if input_vars != set(values['input_variables']):
                raise ValueError(f'Got mismatched input_variables. Expected: {input_vars}. Got: {values['input_variables']}')
        else:
            values['input_variables'] = sorted(input_vars)
        values['input_types'] = input_types
        return values

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate:
        """Create a chat prompt template from a template string.

        Creates a chat template consisting of a single message assumed to be from
        the human.

        Args:
            template: template string
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages([message])

    @classmethod
    @deprecated('0.0.1', alternative='from_messages classmethod', pending=True)
    def from_role_strings(cls, string_messages: List[Tuple[str, str]]) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role, template) tuples.

        Args:
            string_messages: list of (role, template) tuples.

        Returns:
            a chat prompt template
        """
        return cls(messages=[ChatMessagePromptTemplate.from_template(template, role=role) for role, template in string_messages])

    @classmethod
    @deprecated('0.0.1', alternative='from_messages classmethod', pending=True)
    def from_strings(cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]) -> ChatPromptTemplate:
        """Create a chat prompt template from a list of (role class, template) tuples.

        Args:
            string_messages: list of (role class, template) tuples.

        Returns:
            a chat prompt template
        """
        return cls.from_messages(string_messages)

    @classmethod
    def from_messages(cls, messages: Sequence[MessageLikeRepresentation], template_format: Literal['f-string', 'mustache']='f-string') -> ChatPromptTemplate:
        """Create a chat prompt template from a variety of message formats.

        Examples:

            Instantiation from a list of message templates:

            .. code-block:: python

                template = ChatPromptTemplate.from_messages([
                    ("human", "Hello, how are you?"),
                    ("ai", "I'm doing well, thanks!"),
                    ("human", "That's good to hear."),
                ])

            Instantiation from mixed message formats:

            .. code-block:: python

                template = ChatPromptTemplate.from_messages([
                    SystemMessage(content="hello"),
                    ("human", "Hello, how are you?"),
                ])

        Args:
            messages: sequence of message representations.
                  A message can be represented using the following formats:
                  (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
                  (message type, template); e.g., ("human", "{user_input}"),
                  (4) 2-tuple of (message class, template), (4) a string which is
                  shorthand for ("human", template); e.g., "{user_input}"

        Returns:
            a chat prompt template
        """
        _messages = [_convert_to_message(message, template_format) for message in messages]
        input_vars: Set[str] = set()
        partial_vars: Dict[str, Any] = {}
        for _message in _messages:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
            elif isinstance(_message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)):
                input_vars.update(_message.input_variables)
        return cls(input_variables=sorted(input_vars), messages=_messages, partial_variables=partial_vars)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
            else:
                raise ValueError(f'Unexpected input: {message_template}')
        return result

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                message = await message_template.aformat_messages(**kwargs)
                result.extend(message)
            else:
                raise ValueError(f'Unexpected input: {message_template}')
        return result

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """Get a new ChatPromptTemplate with some input variables already filled in.

        Args:
            **kwargs: keyword arguments to use for filling in template variables. Ought
                        to be a subset of the input variables.

        Returns:
            A new ChatPromptTemplate.


        Example:

            .. code-block:: python

                from langchain_core.prompts import ChatPromptTemplate

                template = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are an AI assistant named {name}."),
                        ("human", "Hi I'm {user}"),
                        ("ai", "Hi there, {user}, I'm {name}."),
                        ("human", "{input}"),
                    ]
                )
                template2 = template.partial(user="Lucy", name="R2D2")

                template2.format_messages(input="hello")
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict['input_variables'] = list(set(self.input_variables).difference(kwargs))
        prompt_dict['partial_variables'] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def append(self, message: MessageLikeRepresentation) -> None:
        """Append message to the end of the chat template.

        Args:
            message: representation of a message to append.
        """
        self.messages.append(_convert_to_message(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """Extend the chat template with a sequence of messages."""
        self.messages.extend([_convert_to_message(message) for message in messages])

    @overload
    def __getitem__(self, index: int) -> MessageLike:
        ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[MessageLike, ChatPromptTemplate]:
        """Use to index into the chat template."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return ChatPromptTemplate.from_messages(messages)
        else:
            return self.messages[index]

    def __len__(self) -> int:
        """Get the length of the chat template."""
        return len(self.messages)

    @property
    def _prompt_type(self) -> str:
        """Name of prompt type."""
        return 'chat'

    def save(self, file_path: Union[Path, str]) -> None:
        """Save prompt to file.

        Args:
            file_path: path to file.
        """
        raise NotImplementedError()

    def pretty_repr(self, html: bool=False) -> str:
        return '\n\n'.join((msg.pretty_repr(html=html) for msg in self.messages))