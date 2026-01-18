from __future__ import annotations
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
from langchain.chains.llm import LLMChain
def _do_generation(self, questions: List[str], user_input: str, response: str, _run_manager: CallbackManagerForChainRun) -> Tuple[str, bool]:
    callbacks = _run_manager.get_child()
    docs = []
    for question in questions:
        docs.extend(self.retriever.get_relevant_documents(question))
    context = '\n\n'.join((d.page_content for d in docs))
    result = self.response_chain.predict(user_input=user_input, context=context, response=response, callbacks=callbacks)
    marginal, finished = self.output_parser.parse(result)
    return (marginal, finished)