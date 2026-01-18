from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.messages.ai import (
from langchain_core.messages.base import (
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
def _message_from_dict(message: dict) -> BaseMessage:
    _type = message['type']
    if _type == 'human':
        return HumanMessage(**message['data'])
    elif _type == 'ai':
        return AIMessage(**message['data'])
    elif _type == 'system':
        return SystemMessage(**message['data'])
    elif _type == 'chat':
        return ChatMessage(**message['data'])
    elif _type == 'function':
        return FunctionMessage(**message['data'])
    elif _type == 'tool':
        return ToolMessage(**message['data'])
    elif _type == 'AIMessageChunk':
        return AIMessageChunk(**message['data'])
    elif _type == 'HumanMessageChunk':
        return HumanMessageChunk(**message['data'])
    elif _type == 'FunctionMessageChunk':
        return FunctionMessageChunk(**message['data'])
    elif _type == 'ToolMessageChunk':
        return ToolMessageChunk(**message['data'])
    elif _type == 'SystemMessageChunk':
        return SystemMessageChunk(**message['data'])
    elif _type == 'ChatMessageChunk':
        return ChatMessageChunk(**message['data'])
    else:
        raise ValueError(f'Got unexpected message type: {_type}')