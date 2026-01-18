from typing import Any, Callable, Dict, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_filter_prompt import (
def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(template=prompt_template, input_variables=['question', 'context'], output_parser=BooleanOutputParser())