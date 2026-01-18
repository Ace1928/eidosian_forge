from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
class PairwiseStringDistanceEvalChain(PairwiseStringEvaluator, _RapidFuzzChainMixin):
    """Compute string edit distances between two predictions."""

    @property
    def input_keys(self) -> List[str]:
        """
        Get the input keys.

        Returns:
            List[str]: The input keys.
        """
        return ['prediction', 'prediction_b']

    @property
    def evaluation_name(self) -> str:
        """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
        return f'pairwise_{self.distance.value}_distance'

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """
        Compute the string distance between two predictions.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (CallbackManagerForChainRun , optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {'score': self.compute_metric(inputs['prediction'], inputs['prediction_b'])}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """
        Asynchronously compute the string distance between two predictions.

        Args:
            inputs (Dict[str, Any]): The input values.
            run_manager (AsyncCallbackManagerForChainRun , optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The evaluation results containing the score.
        """
        return {'score': self.compute_metric(inputs['prediction'], inputs['prediction_b'])}

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

    async def _aevaluate_string_pairs(self, *, prediction: str, prediction_b: str, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """
        Asynchronously evaluate the string distance between two predictions.

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
        result = await self.acall(inputs={'prediction': prediction, 'prediction_b': prediction_b}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)