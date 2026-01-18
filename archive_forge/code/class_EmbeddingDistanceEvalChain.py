from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
from langchain.utils.math import cosine_similarity
class EmbeddingDistanceEvalChain(_EmbeddingDistanceChainMixin, StringEvaluator):
    """Use embedding distances to score semantic difference between
    a prediction and reference.

    Examples:
        >>> chain = EmbeddingDistanceEvalChain()
        >>> result = chain.evaluate_strings(prediction="Hello", reference="Hi")
        >>> print(result)
        {'score': 0.5}
    """

    @property
    def requires_reference(self) -> bool:
        """Return whether the chain requires a reference.

        Returns:
            bool: True if a reference is required, False otherwise.
        """
        return True

    @property
    def evaluation_name(self) -> str:
        return f'embedding_{self.distance_metric.value}_distance'

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys of the chain.

        Returns:
            List[str]: The input keys.
        """
        return ['prediction', 'reference']

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Compute the score for a prediction and reference.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (Optional[CallbackManagerForChainRun], optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        vectors = np.array(self.embeddings.embed_documents([inputs['prediction'], inputs['reference']]))
        score = self._compute_score(vectors)
        return {'score': score}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun]=None) -> Dict[str, Any]:
        """Asynchronously compute the score for a prediction and reference.

        Args:
            inputs (Dict[str, Any]): The input data.
            run_manager (AsyncCallbackManagerForChainRun, optional):
                The callback manager.

        Returns:
            Dict[str, Any]: The computed score.
        """
        embedded = await self.embeddings.aembed_documents([inputs['prediction'], inputs['reference']])
        vectors = np.array(embedded)
        score = self._compute_score(vectors)
        return {'score': score}

    def _evaluate_strings(self, *, prediction: str, reference: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Evaluate the embedding distance between a prediction and
        reference.

        Args:
            prediction (str): The output string from the first model.
            reference (str): The reference string (required)
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = self(inputs={'prediction': prediction, 'reference': reference}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)

    async def _aevaluate_strings(self, *, prediction: str, reference: Optional[str]=None, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, include_run_info: bool=False, **kwargs: Any) -> dict:
        """Asynchronously evaluate the embedding distance between
        a prediction and reference.

        Args:
            prediction (str): The output string from the first model.
            reference (str): The output string from the second model.
            callbacks (Callbacks, optional): The callbacks to use.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - score: The embedding distance between the two
                    predictions.
        """
        result = await self.acall(inputs={'prediction': prediction, 'reference': reference}, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=include_run_info)
        return self._prepare_output(result)