from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
def _evaluate_string_pairs(self, *, prediction: str, prediction_b: str, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
    """
        Evaluate the string distance between two predictions.

        Args:
            prediction (str): The first prediction string.
            prediction_b (str): The second prediction string.
            callbacks (Callbacks, optional): The callbacks to use.
            tags (List[str], optional): Tags to apply to traces.
            metadata (Dict[str, Any], optional): Metadata to apply to traces.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The evaluation results containing the score.
        """
    result = self(inputs={'prediction': prediction, 'prediction_b': prediction_b}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
    return self._prepare_output(result)