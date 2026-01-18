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
class ExactMatch(SingleKeyEvalConfig):
    """Configuration for an exact match string evaluator.

        Parameters
        ----------
        ignore_case : bool
            Whether to ignore case when comparing strings.
        ignore_punctuation : bool
            Whether to ignore punctuation when comparing strings.
        ignore_numbers : bool
            Whether to ignore numbers when comparing strings.
        """
    evaluator_type: EvaluatorType = EvaluatorType.EXACT_MATCH
    ignore_case: bool = False
    ignore_punctuation: bool = False
    ignore_numbers: bool = False