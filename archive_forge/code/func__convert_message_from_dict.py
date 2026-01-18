import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@classmethod
def _convert_message_from_dict(cls, message: Dict) -> BaseMessage:
    """Convert a single message from a BaseMessage."""
    role = message['role']
    content = message['content']
    if role == 'user':
        return HumanMessage(content=content)
    elif role == 'assistant':
        return AIMessage(content=content)
    elif role == 'system':
        return SystemMessage(content=content)
    else:
        raise ValueError(f'Got unsupported role: {role}')