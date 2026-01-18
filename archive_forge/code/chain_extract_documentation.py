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
Initialize from LLM.