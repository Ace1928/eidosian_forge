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
class SingleKeyEvalConfig(EvalConfig):
    """Configuration for a run evaluator that only requires a single key."""
    reference_key: Optional[str] = None
    'The key in the dataset run to use as the reference string.\n    If not provided, we will attempt to infer automatically.'
    prediction_key: Optional[str] = None
    "The key from the traced run's outputs dictionary to use to\n    represent the prediction. If not provided, it will be inferred\n    automatically."
    input_key: Optional[str] = None
    "The key from the traced run's inputs dictionary to use to represent the\n    input. If not provided, it will be inferred automatically."

    def get_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_kwargs()
        for key in ['reference_key', 'prediction_key', 'input_key']:
            kwargs.pop(key, None)
        return kwargs