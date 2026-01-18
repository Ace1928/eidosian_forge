from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from langchain.evaluation.criteria.eval_chain import CRITERIA_TYPE
from langchain.evaluation.embedding_distance.base import (
from langchain.evaluation.schema import EvaluatorType, StringEvaluator
from langchain.evaluation.string_distance.base import (
class ScoreString(SingleKeyEvalConfig):
    """Configuration for a score string evaluator.
        This is like the criteria evaluator but it is configured by
        default to return a score on the scale from 1-10.

        It is recommended to normalize these scores
        by setting `normalize_by` to 10.

        Parameters
        ----------
        criteria : Optional[CRITERIA_TYPE]
            The criteria to evaluate.
        llm : Optional[BaseLanguageModel]
            The language model to use for the evaluation chain.
        normalize_by: Optional[int] = None
            If you want to normalize the score, the denominator to use.
            If not provided, the score will be between 1 and 10 (by default).
        prompt : Optional[BasePromptTemplate]

        """
    evaluator_type: EvaluatorType = EvaluatorType.SCORE_STRING
    criteria: Optional[CRITERIA_TYPE] = None
    llm: Optional[BaseLanguageModel] = None
    normalize_by: Optional[float] = None
    prompt: Optional[BasePromptTemplate] = None

    def __init__(self, criteria: Optional[CRITERIA_TYPE]=None, normalize_by: Optional[float]=None, **kwargs: Any) -> None:
        super().__init__(criteria=criteria, normalize_by=normalize_by, **kwargs)