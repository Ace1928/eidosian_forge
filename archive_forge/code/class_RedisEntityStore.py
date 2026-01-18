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
class RedisEntityStore(BaseEntityStore):
    """Redis-backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.
    """
    redis_client: Any
    session_id: str = 'default'
    key_prefix: str = 'memory_store'
    ttl: Optional[int] = 60 * 60 * 24
    recall_ttl: Optional[int] = 60 * 60 * 24 * 3

    def __init__(self, session_id: str='default', url: str='redis://localhost:6379/0', key_prefix: str='memory_store', ttl: Optional[int]=60 * 60 * 24, recall_ttl: Optional[int]=60 * 60 * 24 * 3, *args: Any, **kwargs: Any):
        try:
            import redis
        except ImportError:
            raise ImportError('Could not import redis python package. Please install it with `pip install redis`.')
        super().__init__(*args, **kwargs)
        try:
            self.redis_client = get_client(redis_url=url, decode_responses=True)
        except redis.exceptions.ConnectionError as error:
            logger.error(error)
        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        return f'{self.key_prefix}:{self.session_id}'

    def get(self, key: str, default: Optional[str]=None) -> Optional[str]:
        res = self.redis_client.getex(f'{self.full_key_prefix}:{key}', ex=self.recall_ttl) or default or ''
        logger.debug(f"REDIS MEM get '{self.full_key_prefix}:{key}': '{res}'")
        return res

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f'{self.full_key_prefix}:{key}', value, ex=self.ttl)
        logger.debug(f"REDIS MEM set '{self.full_key_prefix}:{key}': '{value}' EX {self.ttl}")

    def delete(self, key: str) -> None:
        self.redis_client.delete(f'{self.full_key_prefix}:{key}')

    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f'{self.full_key_prefix}:{key}') == 1

    def clear(self) -> None:

        def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[Any]:
            iterator = iter(iterable)
            while (batch := list(islice(iterator, batch_size))):
                yield batch
        for keybatch in batched(self.redis_client.scan_iter(f'{self.full_key_prefix}:*'), 500):
            self.redis_client.delete(*keybatch)