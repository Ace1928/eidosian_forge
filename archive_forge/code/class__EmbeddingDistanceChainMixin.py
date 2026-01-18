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
class _EmbeddingDistanceChainMixin(Chain):
    """Shared functionality for embedding distance evaluators.

    Attributes:
        embeddings (Embeddings): The embedding objects to vectorize the outputs.
        distance_metric (EmbeddingDistance): The distance metric to use
                                            for comparing the embeddings.
    """
    embeddings: Embeddings = Field(default_factory=OpenAIEmbeddings)
    distance_metric: EmbeddingDistance = Field(default=EmbeddingDistance.COSINE)

    @root_validator(pre=False)
    def _validate_tiktoken_installed(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the TikTok library is installed.

        Args:
            values (Dict[str, Any]): The values to validate.

        Returns:
            Dict[str, Any]: The validated values.
        """
        embeddings = values.get('embeddings')
        if isinstance(embeddings, OpenAIEmbeddings):
            try:
                import tiktoken
            except ImportError:
                raise ImportError('The tiktoken library is required to use the default OpenAI embeddings with embedding distance evaluators. Please either manually select a different Embeddings object or install tiktoken using `pip install tiktoken`.')
        return values

    class Config:
        """Permit embeddings to go unvalidated."""
        arbitrary_types_allowed: bool = True

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys of the chain.

        Returns:
            List[str]: The output keys.
        """
        return ['score']

    def _prepare_output(self, result: dict) -> dict:
        parsed = {'score': result['score']}
        if RUN_KEY in result:
            parsed[RUN_KEY] = result[RUN_KEY]
        return parsed

    def _get_metric(self, metric: EmbeddingDistance) -> Any:
        """Get the metric function for the given metric name.

        Args:
            metric (EmbeddingDistance): The metric name.

        Returns:
            Any: The metric function.
        """
        metrics = {EmbeddingDistance.COSINE: self._cosine_distance, EmbeddingDistance.EUCLIDEAN: self._euclidean_distance, EmbeddingDistance.MANHATTAN: self._manhattan_distance, EmbeddingDistance.CHEBYSHEV: self._chebyshev_distance, EmbeddingDistance.HAMMING: self._hamming_distance}
        if metric in metrics:
            return metrics[metric]
        else:
            raise ValueError(f'Invalid metric: {metric}')

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the cosine distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.ndarray: The cosine distance.
        """
        return 1.0 - cosine_similarity(a, b)

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Euclidean distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Euclidean distance.
        """
        return np.linalg.norm(a - b)

    @staticmethod
    def _manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Manhattan distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Manhattan distance.
        """
        return np.sum(np.abs(a - b))

    @staticmethod
    def _chebyshev_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Chebyshev distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Chebyshev distance.
        """
        return np.max(np.abs(a - b))

    @staticmethod
    def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
        """Compute the Hamming distance between two vectors.

        Args:
            a (np.ndarray): The first vector.
            b (np.ndarray): The second vector.

        Returns:
            np.floating: The Hamming distance.
        """
        return np.mean(a != b)

    def _compute_score(self, vectors: np.ndarray) -> float:
        """Compute the score based on the distance metric.

        Args:
            vectors (np.ndarray): The input vectors.

        Returns:
            float: The computed score.
        """
        metric = self._get_metric(self.distance_metric)
        score = metric(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1)).item()
        return score