from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
def _get_chat_history(chat_history: List[CHAT_TURN_TYPE]) -> str:
    buffer = ''
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f'{dialogue_turn.type}: ')
            buffer += f'\n{role_prefix}{dialogue_turn.content}'
        elif isinstance(dialogue_turn, tuple):
            human = 'Human: ' + dialogue_turn[0]
            ai = 'Assistant: ' + dialogue_turn[1]
            buffer += '\n' + '\n'.join([human, ai])
        else:
            raise ValueError(f'Unsupported chat history format: {type(dialogue_turn)}. Full chat history: {chat_history} ')
    return buffer