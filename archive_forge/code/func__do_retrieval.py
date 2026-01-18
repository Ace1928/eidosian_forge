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
def _do_retrieval(self, low_confidence_spans: List[str], _run_manager: CallbackManagerForChainRun, user_input: str, response: str, initial_response: str) -> Tuple[str, bool]:
    question_gen_inputs = [{'user_input': user_input, 'current_response': initial_response, 'uncertain_span': span} for span in low_confidence_spans]
    callbacks = _run_manager.get_child()
    question_gen_outputs = self.question_generator_chain.apply(question_gen_inputs, callbacks=callbacks)
    questions = [output[self.question_generator_chain.output_keys[0]] for output in question_gen_outputs]
    _run_manager.on_text(f'Generated Questions: {questions}', color='yellow', end='\n')
    return self._do_generation(questions, user_input, response, _run_manager)