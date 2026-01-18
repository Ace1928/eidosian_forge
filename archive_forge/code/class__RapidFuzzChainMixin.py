from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
class _RapidFuzzChainMixin(Chain):
    """Shared methods for the rapidfuzz string distance evaluators."""
    distance: StringDistance = Field(default=StringDistance.JARO_WINKLER)
    normalize_score: bool = Field(default=True)
    'Whether to normalize the score to a value between 0 and 1.\n    Applies only to the Levenshtein and Damerau-Levenshtein distances.'

    @root_validator
    def validate_dependencies(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the rapidfuzz library is installed.

        Args:
            values (Dict[str, Any]): The input values.

        Returns:
            Dict[str, Any]: The validated values.
        """
        _load_rapidfuzz()
        return values

    @property
    def output_keys(self) -> List[str]:
        """
        Get the output keys.

        Returns:
            List[str]: The output keys.
        """
        return ['score']

    def _prepare_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the output dictionary.

        Args:
            result (Dict[str, Any]): The evaluation results.

        Returns:
            Dict[str, Any]: The prepared output dictionary.
        """
        result = {'score': result['score']}
        if RUN_KEY in result:
            result[RUN_KEY] = result[RUN_KEY].dict()
        return result

    @staticmethod
    def _get_metric(distance: str, normalize_score: bool=False) -> Callable:
        """
        Get the distance metric function based on the distance type.

        Args:
            distance (str): The distance type.

        Returns:
            Callable: The distance metric function.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        from rapidfuzz import distance as rf_distance
        module_map: Dict[str, Any] = {StringDistance.DAMERAU_LEVENSHTEIN: rf_distance.DamerauLevenshtein, StringDistance.LEVENSHTEIN: rf_distance.Levenshtein, StringDistance.JARO: rf_distance.Jaro, StringDistance.JARO_WINKLER: rf_distance.JaroWinkler, StringDistance.HAMMING: rf_distance.Hamming, StringDistance.INDEL: rf_distance.Indel}
        if distance not in module_map:
            raise ValueError(f'Invalid distance metric: {distance}\nMust be one of: {list(StringDistance)}')
        module = module_map[distance]
        if normalize_score:
            return module.normalized_distance
        else:
            return module.distance

    @property
    def metric(self) -> Callable:
        """
        Get the distance metric function.

        Returns:
            Callable: The distance metric function.
        """
        return _RapidFuzzChainMixin._get_metric(self.distance, normalize_score=self.normalize_score)

    def compute_metric(self, a: str, b: str) -> float:
        """
        Compute the distance between two strings.

        Args:
            a (str): The first string.
            b (str): The second string.

        Returns:
            float: The distance between the two strings.
        """
        return self.metric(a, b)