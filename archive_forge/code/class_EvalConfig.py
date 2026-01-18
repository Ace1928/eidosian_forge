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
class EvalConfig(BaseModel):
    """Configuration for a given run evaluator.

    Parameters
    ----------
    evaluator_type : EvaluatorType
        The type of evaluator to use.

    Methods
    -------
    get_kwargs()
        Get the keyword arguments for the evaluator configuration.

    """
    evaluator_type: EvaluatorType

    def get_kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments for the load_evaluator call.

        Returns
        -------
        Dict[str, Any]
            The keyword arguments for the load_evaluator call.

        """
        kwargs = {}
        for field, val in self:
            if field == 'evaluator_type':
                continue
            elif val is None:
                continue
            kwargs[field] = val
        return kwargs