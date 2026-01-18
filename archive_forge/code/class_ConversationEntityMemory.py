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
class ConversationEntityMemory(BaseChatMemory):
    """Entity extractor & summarizer memory.

    Extracts named entities from the recent chat history and generates summaries.
    With a swappable entity store, persisting entities across conversations.
    Defaults to an in-memory entity store, and can be swapped out for a Redis,
    SQLite, or other entity store.
    """
    human_prefix: str = 'Human'
    ai_prefix: str = 'AI'
    llm: BaseLanguageModel
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT
    entity_cache: List[str] = []
    k: int = 3
    chat_history_key: str = 'history'
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)

    @property
    def buffer(self) -> List[BaseMessage]:
        """Access chat memory messages."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return ['entities', self.chat_history_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns chat history and all generated entities with summaries if available,
        and updates or clears the recent entity cache.

        New entity name can be found when calling this method, before the entity
        summaries are generated, so the entity cache values may be empty if no entity
        descriptions are generated yet.
        """
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        buffer_string = get_buffer_string(self.buffer[-self.k * 2:], human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
        output = chain.predict(history=buffer_string, input=inputs[prompt_input_key])
        if output.strip() == 'NONE':
            entities = []
        else:
            entities = [w.strip() for w in output.split(',')]
        entity_summaries = {}
        for entity in entities:
            entity_summaries[entity] = self.entity_store.get(entity, '')
        self.entity_cache = entities
        if self.return_messages:
            buffer: Any = self.buffer[-self.k * 2:]
        else:
            buffer = buffer_string
        return {self.chat_history_key: buffer, 'entities': entity_summaries}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context from this conversation history to the entity store.

        Generates a summary for each entity in the entity cache by prompting
        the model, and saves these summaries to the entity store.
        """
        super().save_context(inputs, outputs)
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        buffer_string = get_buffer_string(self.buffer[-self.k * 2:], human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
        input_data = inputs[prompt_input_key]
        chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)
        for entity in self.entity_cache:
            existing_summary = self.entity_store.get(entity, '')
            output = chain.predict(summary=existing_summary, entity=entity, history=buffer_string, input=input_data)
            self.entity_store.set(entity, output.strip())

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        self.entity_cache.clear()
        self.entity_store.clear()