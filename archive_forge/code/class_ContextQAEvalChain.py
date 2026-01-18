from __future__ import annotations
import re
import string
from typing import Any, List, Optional, Sequence, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import CONTEXT_PROMPT, COT_PROMPT, PROMPT
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY
class ContextQAEvalChain(LLMChain, StringEvaluator, LLMEvalChain):
    """LLM Chain for evaluating QA w/o GT based on context"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """Whether the chain requires a reference string."""
        return True

    @property
    def requires_input(self) -> bool:
        """Whether the chain requires an input string."""
        return True

    class Config:
        """Configuration for the QAEvalChain."""
        extra = Extra.ignore

    @classmethod
    def _validate_input_vars(cls, prompt: PromptTemplate) -> None:
        expected_input_vars = {'query', 'context', 'result'}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt.input_variables}')

    @property
    def evaluation_name(self) -> str:
        return 'Contextual Accuracy'

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: Optional[PromptTemplate]=None, **kwargs: Any) -> ContextQAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm (BaseLanguageModel): the base language model to use.

            prompt (PromptTemplate): A prompt template containing the input_variables:
            'query', 'context' and 'result' that will be used as the prompt
            for evaluation.
            Defaults to PROMPT.

            **kwargs: additional keyword arguments.

        Returns:
            ContextQAEvalChain: the loaded QA eval chain.
        """
        prompt = prompt or CONTEXT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(self, examples: List[dict], predictions: List[dict], question_key: str='query', context_key: str='context', prediction_key: str='result', *, callbacks: Callbacks=None) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = [{'query': example[question_key], 'context': example[context_key], 'result': predictions[i][prediction_key]} for i, example in enumerate(examples)]
        return self.apply(inputs, callbacks=callbacks)

    def _prepare_output(self, result: dict) -> dict:
        parsed_result = _parse_string_eval_output(result[self.output_key])
        if RUN_KEY in result:
            parsed_result[RUN_KEY] = result[RUN_KEY]
        return parsed_result

    def _evaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        result = self({'query': input, 'context': reference, 'result': prediction}, callbacks=callbacks, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        result = await self.acall(inputs={'query': input, 'context': reference, 'result': prediction}, callbacks=callbacks, include_run_info=include_run_info)
        return self._prepare_output(result)