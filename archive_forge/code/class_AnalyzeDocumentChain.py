from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.base import Chain
class AnalyzeDocumentChain(Chain):
    """Chain that splits documents, then analyzes it in pieces.

    This chain is parameterized by a TextSplitter and a CombineDocumentsChain.
    This chain takes a single document as input, and then splits it up into chunks
    and then passes those chucks to the CombineDocumentsChain.
    """
    input_key: str = 'input_document'
    text_splitter: TextSplitter = Field(default_factory=RecursiveCharacterTextSplitter)
    combine_docs_chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return self.combine_docs_chain.output_keys

    def get_input_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        return create_model('AnalyzeDocumentChain', **{self.input_key: (str, None)})

    def get_output_schema(self, config: Optional[RunnableConfig]=None) -> Type[BaseModel]:
        return self.combine_docs_chain.get_output_schema(config)

    def _call(self, inputs: Dict[str, str], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, str]:
        """Split document into chunks and pass to CombineDocumentsChain."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        other_keys: Dict = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(other_keys, return_only_outputs=True, callbacks=_run_manager.get_child())