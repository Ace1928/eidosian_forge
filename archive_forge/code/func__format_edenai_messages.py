import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from aiohttp import ClientSession
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.utilities.requests import Requests
def _format_edenai_messages(messages: List[BaseMessage]) -> Dict[str, Any]:
    system = None
    formatted_messages = []
    text = messages[-1].content
    for i, message in enumerate(messages[:-1]):
        if message.type == 'system':
            if i != 0:
                raise ValueError('System message must be at beginning of message list.')
            system = message.content
        else:
            formatted_messages.append({'role': _message_role(message.type), 'message': message.content})
    return {'text': text, 'previous_history': formatted_messages, 'chatbot_global_action': system}