from __future__ import annotations
import asyncio
from typing import Any, Callable, Dict, Optional, Sequence, cast
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_extract_prompt import (
class NoOutputParser(BaseOutputParser[str]):
    """Parse outputs that could return a null string of some sort."""
    no_output_str: str = 'NO_OUTPUT'

    def parse(self, text: str) -> str:
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ''
        return cleaned_text