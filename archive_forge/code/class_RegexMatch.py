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
class RegexMatch(SingleKeyEvalConfig):
    """Configuration for a regex match string evaluator.

        Parameters
        ----------
        flags : int
            The flags to pass to the regex. Example: re.IGNORECASE.
        """
    evaluator_type: EvaluatorType = EvaluatorType.REGEX_MATCH
    flags: int = 0