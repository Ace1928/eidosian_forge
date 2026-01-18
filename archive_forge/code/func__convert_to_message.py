from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.messages.ai import (
from langchain_core.messages.base import (
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
    - dict: a message dict with role and content keys
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats

    Returns:
        an instance of a message or a message template
    """
    if isinstance(message, BaseMessage):
        _message = message
    elif isinstance(message, str):
        _message = _create_message_from_message_type('human', message)
    elif isinstance(message, tuple):
        if len(message) != 2:
            raise ValueError(f'Expected 2-tuple of (role, template), got {message}')
        message_type_str, template = message
        _message = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
        msg_kwargs = message.copy()
        try:
            msg_type = msg_kwargs.pop('role')
            msg_content = msg_kwargs.pop('content')
        except KeyError:
            raise ValueError(f"Message dict must contain 'role' and 'content' keys, got {message}")
        _message = _create_message_from_message_type(msg_type, msg_content, **msg_kwargs)
    else:
        raise NotImplementedError(f'Unsupported message type: {type(message)}')
    return _message