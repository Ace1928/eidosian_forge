from typing import Any, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.llms.mlx_pipeline import MLXPipeline
def _to_chatml_format(self, message: BaseMessage) -> dict:
    """Convert LangChain message to ChatML format."""
    if isinstance(message, SystemMessage):
        role = 'system'
    elif isinstance(message, AIMessage):
        role = 'assistant'
    elif isinstance(message, HumanMessage):
        role = 'user'
    else:
        raise ValueError(f'Unknown message type: {type(message)}')
    return {'role': role, 'content': message.content}