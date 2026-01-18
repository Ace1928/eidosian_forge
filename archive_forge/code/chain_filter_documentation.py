from typing import Any, Callable, Dict, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_filter_prompt import (
Create a LLMChainFilter from a language model.

        Args:
            llm: The language model to use for filtering.
            prompt: The prompt to use for the filter.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A LLMChainFilter that uses the given language model.
        