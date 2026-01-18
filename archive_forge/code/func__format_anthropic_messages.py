import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra
from langchain_community.chat_models.anthropic import (
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama
from langchain_community.llms.bedrock import BedrockBase
from langchain_community.utilities.anthropic import (
def _format_anthropic_messages(messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for anthropic."""
    '\n    [\n        {\n            "role": _message_type_lookups[m.type],\n            "content": [_AnthropicMessageContent(text=m.content).dict()],\n        }\n        for m in messages\n    ]\n    '
    system: Optional[str] = None
    formatted_messages: List[Dict] = []
    for i, message in enumerate(messages):
        if message.type == 'system':
            if i != 0:
                raise ValueError('System message must be at beginning of message list.')
            if not isinstance(message.content, str):
                raise ValueError(f'System message must be a string, instead was: {type(message.content)}')
            system = message.content
            continue
        role = _message_type_lookups[message.type]
        content: Union[str, List[Dict]]
        if not isinstance(message.content, str):
            assert isinstance(message.content, list), 'Anthropic message content must be str or list of dicts'
            content = []
            for item in message.content:
                if isinstance(item, str):
                    content.append({'type': 'text', 'text': item})
                elif isinstance(item, dict):
                    if 'type' not in item:
                        raise ValueError('Dict content item must have a type key')
                    if item['type'] == 'image_url':
                        source = _format_image(item['image_url']['url'])
                        content.append({'type': 'image', 'source': source})
                    else:
                        content.append(item)
                else:
                    raise ValueError(f'Content items must be str or dict, instead was: {type(item)}')
        else:
            content = message.content
        formatted_messages.append({'role': role, 'content': content})
    return (system, formatted_messages)