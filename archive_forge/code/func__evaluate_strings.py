from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
def _evaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
    """
        Evaluate the string distance between the prediction and the reference.

        Args:
            prediction (str): The prediction string.
            reference (Optional[str], optional): The reference string.
            input (Optional[str], optional): The input string.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
    result = self(inputs={'prediction': prediction, 'reference': reference}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
    return self._prepare_output(result)