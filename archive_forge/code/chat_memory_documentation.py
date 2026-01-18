import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field
from langchain.memory.utils import get_prompt_input_key
Clear memory contents.