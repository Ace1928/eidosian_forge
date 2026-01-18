from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
class _Expect:
    """A class for setting expectations on test results."""

    def __init__(self, *, client: Optional[ls_client.Client]=None):
        self.client = client or ls_client.Client()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        atexit.register(self.executor.shutdown, wait=True)

    def embedding_distance(self, prediction: str, reference: str, *, config: Optional[EmbeddingConfig]=None) -> _Matcher:
        """Compute the embedding distance between the prediction and reference.

        This logs the embedding distance to LangSmith and returns a `_Matcher` instance
        for making assertions on the distance value.

        By default, this uses the OpenAI API for computing embeddings.

        Args:
            prediction: The predicted string to compare.
            reference: The reference string to compare against.
            config: Optional configuration for the embedding distance evaluator.
                Supported options:
                - `encoder`: A custom encoder function to encode the list of input
                     strings to embeddings. Defaults to the OpenAI API.
                - `metric`: The distance metric to use for comparison.
                    Supported values: "cosine", "euclidean", "manhattan",
                    "chebyshev", "hamming".

        Returns:
            A `_Matcher` instance for the embedding distance value.


        Examples:
            >>> expect.embedding_distance(
            ...     prediction="hello",
            ...     reference="hi",
            ... ).to_be_less_than(1.0)
        """
        from langsmith._internal._embedding_distance import EmbeddingDistance
        config = config or {}
        encoder_func = 'custom' if config.get('encoder') else 'openai'
        evaluator = EmbeddingDistance(config=config)
        score = evaluator.evaluate(prediction=prediction, reference=reference)
        src_info = {'encoder': encoder_func, 'metric': evaluator.distance}
        self._submit_feedback('embedding_distance', {'score': score, 'source_info': src_info, 'comment': f'Using {encoder_func}, Metric: {evaluator.distance}'})
        return _Matcher(self.client, 'embedding_distance', score, _executor=self.executor)

    def edit_distance(self, prediction: str, reference: str, *, config: Optional[EditDistanceConfig]=None) -> _Matcher:
        """Compute the string distance between the prediction and reference.

        This logs the string distance (Damerau-Levenshtein) to LangSmith and returns
        a `_Matcher` instance for making assertions on the distance value.

        This depends on the `rapidfuzz` package for string distance computation.

        Args:
            prediction: The predicted string to compare.
            reference: The reference string to compare against.
            config: Optional configuration for the string distance evaluator.
                Supported options:
                - `metric`: The distance metric to use for comparison.
                    Supported values: "damerau_levenshtein", "levenshtein",
                    "jaro", "jaro_winkler", "hamming", "indel".
                - `normalize_score`: Whether to normalize the score between 0 and 1.

        Returns:
            A `_Matcher` instance for the string distance value.

        Examples:
            >>> expect.edit_distance("hello", "helo").to_be_less_than(1)
        """
        from langsmith._internal._edit_distance import EditDistance
        config = config or {}
        metric = config.get('metric') or 'damerau_levenshtein'
        normalize = config.get('normalize_score', True)
        evaluator = EditDistance(config=config)
        score = evaluator.evaluate(prediction=prediction, reference=reference)
        src_info = {'metric': metric, 'normalize': normalize}
        self._submit_feedback('edit_distance', {'score': score, 'source_info': src_info, 'comment': f'Using {metric}, Normalize: {normalize}'})
        return _Matcher(self.client, 'edit_distance', score, _executor=self.executor)

    def value(self, value: Any) -> _Matcher:
        """Create a `_Matcher` instance for making assertions on the given value.

        Args:
            value: The value to make assertions on.

        Returns:
            A `_Matcher` instance for the given value.

        Examples:
           >>> expect.value(10).to_be_less_than(20)
        """
        return _Matcher(self.client, 'value', value, _executor=self.executor)

    @overload
    def __call__(self, value: Any, /) -> _Matcher:
        ...

    @overload
    def __call__(self, /, *, client: ls_client.Client) -> _Expect:
        ...

    def __call__(self, value: Optional[Any]=None, /, client: Optional[ls_client.Client]=None) -> Union[_Expect, _Matcher]:
        expected = _Expect(client=client)
        if value is not None:
            return expected.value(value)
        return expected

    def _submit_feedback(self, key: str, results: dict):
        current_run = rh.get_current_run_tree()
        run_id = current_run.id if current_run else None
        if not ls_utils.test_tracking_is_disabled():
            self.executor.submit(self.client.create_feedback, run_id=run_id, key=key, **results)