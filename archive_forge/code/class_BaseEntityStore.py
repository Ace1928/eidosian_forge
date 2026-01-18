import logging
from abc import ABC, abstractmethod
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional
from langchain_community.utilities.redis import get_client
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
from langchain.memory.utils import get_prompt_input_key
class BaseEntityStore(BaseModel, ABC):
    """Abstract base class for Entity store."""

    @abstractmethod
    def get(self, key: str, default: Optional[str]=None) -> Optional[str]:
        """Get entity value from store."""
        pass

    @abstractmethod
    def set(self, key: str, value: Optional[str]) -> None:
        """Set entity value in store."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete entity value from store."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if entity exists in store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all entities from store."""
        pass