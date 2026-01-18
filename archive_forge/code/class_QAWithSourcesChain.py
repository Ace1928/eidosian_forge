from __future__ import annotations
import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
class QAWithSourcesChain(BaseQAWithSourcesChain):
    """Question answering with sources over documents."""
    input_docs_key: str = 'docs'

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_docs_key, self.question_key]

    def _get_docs(self, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    async def _aget_docs(self, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """Get docs to run questioning over."""
        return inputs.pop(self.input_docs_key)

    @property
    def _chain_type(self) -> str:
        return 'qa_with_sources_chain'