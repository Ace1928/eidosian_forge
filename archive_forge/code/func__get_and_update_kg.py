from typing import Any, Dict, List, Type, Union
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.networkx_graph import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
from langchain.memory.utils import get_prompt_input_key
def _get_and_update_kg(self, inputs: Dict[str, Any]) -> None:
    """Get and update knowledge graph from the conversation history."""
    prompt_input_key = self._get_prompt_input_key(inputs)
    knowledge = self.get_knowledge_triplets(inputs[prompt_input_key])
    for triple in knowledge:
        self.kg.add_triple(triple)