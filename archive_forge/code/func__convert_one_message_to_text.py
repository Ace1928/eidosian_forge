from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_community.llms.anthropic import _AnthropicCommon
def _convert_one_message_to_text(message: BaseMessage, human_prompt: str, ai_prompt: str) -> str:
    content = cast(str, message.content)
    if isinstance(message, ChatMessage):
        message_text = f'\n\n{message.role.capitalize()}: {content}'
    elif isinstance(message, HumanMessage):
        message_text = f'{human_prompt} {content}'
    elif isinstance(message, AIMessage):
        message_text = f'{ai_prompt} {content}'
    elif isinstance(message, SystemMessage):
        message_text = content
    else:
        raise ValueError(f'Got unknown type {message}')
    return message_text