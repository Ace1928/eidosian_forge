from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, Union
from langsmith.evaluation.evaluator import run_evaluator
from langsmith.run_helpers import traceable
from langsmith.schemas import Example, Run
class SingleEvaluatorInput(TypedDict):
    """The input to a `StringEvaluator`."""
    prediction: str
    'The prediction string.'
    reference: Optional[Any]
    'The reference string.'
    input: Optional[str]
    'The input string.'