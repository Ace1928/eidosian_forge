from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.evaluation.scoring.prompt import (
from langchain.schema import RUN_KEY
class ScoreStringEvalChain(StringEvaluator, LLMEvalChain, LLMChain):
    """A chain for scoring on a scale of 1-10 the output of a model.

    Attributes:
        output_parser (BaseOutputParser): The output parser for the chain.

    Example:
        >>> from langchain_community.chat_models import ChatOpenAI
        >>> from langchain.evaluation.scoring import ScoreStringEvalChain
        >>> llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        >>> chain = ScoreStringEvalChain.from_llm(llm=llm)
        >>> result = chain.evaluate_strings(
        ...     input = "What is the chemical formula for water?",
        ...     prediction = "H2O",
        ...     reference = "The chemical formula for water is H2O.",
        ... )
        >>> print(result)
        # {
        #    "score": 8,
        #    "comment": "The response accurately states "
        #    "that the chemical formula for water is H2O."
        #    "However, it does not provide an explanation of what the formula means."
        # }

    """
    output_key: str = 'results'
    output_parser: BaseOutputParser = Field(default_factory=ScoreStringResultOutputParser)
    normalize_by: Optional[float] = None
    'The value to normalize the score by, if specified.'
    criterion_name: str
    'The name of the criterion being evaluated.'

    class Config:
        """Configuration for the ScoreStringEvalChain."""
        extra = Extra.ignore

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if the chain requires a reference, False otherwise.

        """
        return False

    @property
    def requires_input(self) -> bool:
        """Return whether the chain requires an input.

        Returns:
            bool: True if the chain requires an input, False otherwise.

        """
        return True

    @property
    def evaluation_name(self) -> str:
        """Get the name of the evaluation.

        Returns
        -------
        str
            The name of the evaluation.
        """
        return f'score_string:{self.criterion_name}'

    @property
    def _skip_reference_warning(self) -> str:
        """Return the warning to show when reference is ignored.

        Returns:
            str: The warning to show when reference is ignored.

        """
        return f'Ignoring reference in {self.__class__.__name__}, as it is not expected.\nTo use a reference, use the LabeledScoreStringEvalChain instead. (EvaluatorType.LABELED_SCORE_STRING) instead.'

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, *, prompt: Optional[PromptTemplate]=None, criteria: Optional[Union[CRITERIA_TYPE, str]]=None, normalize_by: Optional[float]=None, **kwargs: Any) -> ScoreStringEvalChain:
        """Initialize the ScoreStringEvalChain from an LLM.

        Args:
            llm (BaseChatModel): The LLM to use (GPT-4 recommended).
            prompt (PromptTemplate, optional): The prompt to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ScoreStringEvalChain: The initialized ScoreStringEvalChain.

        Raises:
            ValueError: If the input variables are not as expected.

        """
        if not (isinstance(llm, (ChatOpenAI, AzureChatOpenAI)) and llm.model_name.startswith('gpt-4')):
            logger.warning('This chain was only tested with GPT-4. Performance may be significantly worse with other models.')
        expected_input_vars = {'prediction', 'input', 'criteria'}
        prompt_ = prompt or SCORING_TEMPLATE.partial(reference='')
        if expected_input_vars != set(prompt_.input_variables):
            raise ValueError(f'Input variables should be {expected_input_vars}, but got {prompt_.input_variables}')
        criteria_ = resolve_criteria(criteria)
        criteria_str = '\n'.join((f'{k}: {v}' if v else k for k, v in criteria_.items())).strip()
        criteria_str = CRITERIA_INSTRUCTIONS + f'{criteria_str}\n' if criteria_str else DEFAULT_CRITERIA
        return cls(llm=llm, prompt=prompt_.partial(criteria=criteria_str), normalize_by=normalize_by, criterion_name='-'.join(criteria_), **kwargs)

    def _prepare_input(self, prediction: str, input: Optional[str], reference: Optional[str]) -> dict:
        """Prepare the input for the chain.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            input (str, optional): The input or task string.
            reference (str, optional): The reference string, if any.

        Returns:
            dict: The prepared input for the chain.

        """
        input_ = {'prediction': prediction, 'input': input}
        if self.requires_reference:
            input_['reference'] = reference
        return input_

    def _prepare_output(self, result: dict) -> dict:
        """Prepare the output."""
        parsed = result[self.output_key]
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        if 'score' in parsed and self.normalize_by is not None:
            parsed['score'] = parsed['score'] / self.normalize_by
        return parsed

    def _evaluate_strings(self, *, prediction: str, input: Optional[str]=None, reference: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Score the output string.

        Args:
            prediction (str): The output string from the first model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - score: A score between 1 and 10.

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = self(inputs=input_, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_string_pairs(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Asynchronously score the output string.

        Args:
            prediction (str): The output string from the first model.
            input (str, optional): The input or task string.
            callbacks (Callbacks, optional): The callbacks to use.
            reference (str, optional): The reference string, if any.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - reasoning: The reasoning for the preference.
                - score: A score between 1 and 10.

        """
        input_ = self._prepare_input(prediction, input, reference)
        result = await self.acall(inputs=input_, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)