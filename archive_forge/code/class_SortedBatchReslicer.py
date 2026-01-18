import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
class SortedBatchReslicer(Generic[T]):
    """Reslice batch streams (that are alredy sorted by keys) by keys.

    :param keys: group keys to reslice by
    """

    def __init__(self, keys: List[str]) -> None:
        self._keys = keys
        self._last_row: Optional[np.ndarray] = None

    def take(self, batch: T, start: int, length: int) -> T:
        """Take a slice of the batch

        :param batch: the batch object
        :param start: the start row index
        :param length: the number of rows to take

        :return: a slice of the batch
        """
        raise NotImplementedError

    def concat(self, batches: List[T]) -> T:
        """Concatenate a list of batches into one batch

        :param batches: the list of batches
        :return: the concatenated batch
        """
        raise NotImplementedError

    def get_keys_ndarray(self, batch: T, keys: List[str]) -> np.ndarray:
        """Get the keys as a numpy array

        :param batch: the batch object
        :param keys: the keys to get

        :return: the keys as a numpy array
        """
        raise NotImplementedError

    def get_batch_length(self, batch: T) -> int:
        """Get the number of rows in the batch

        :param batch: the batch object

        :return: the number of rows in the batch
        """
        raise NotImplementedError

    def reslice(self, batches: Iterable[T]) -> Iterable[Iterable[T]]:
        """Reslice the batch stream into a stream of iterable of batches of the
        same keys

        :param batches: the batch stream

        :yield: an iterable of iterable of batches containing same keys
        """

        def slicer(n: int, current: Tuple[bool, T], last: Optional[Tuple[bool, T]]) -> bool:
            return current[0]

        def get_slices() -> Iterable[Tuple[bool, T]]:
            for batch in batches:
                if self.get_batch_length(batch) > 0:
                    yield from self._reslice_single(batch)

        def transform(data: Iterable[Tuple[bool, T]]) -> Iterable[T]:
            for _, batch in data:
                yield batch
        for res in slice_iterable(get_slices(), slicer):
            yield transform(res)

    def reslice_and_merge(self, batches: Iterable[T]) -> Iterable[T]:
        """Reslice the batch stream into new batches, each containing the same keys

        :param batches: the batch stream

        :yield: an iterable of batches, each containing the same keys
        """
        cache: Optional[T] = None
        for batch in batches:
            if self.get_batch_length(batch) > 0:
                for diff, sub in self._reslice_single(batch):
                    if not diff:
                        cache = self.concat([cache, sub])
                    else:
                        if cache is not None:
                            yield cache
                        cache = sub
        if cache is not None:
            yield cache

    def _reslice_single(self, batch: T) -> Iterable[Tuple[bool, T]]:
        a = self.get_keys_ndarray(batch, self._keys)
        b = np.roll(a, 1, axis=0)
        diff = self._diff(a, b)
        if self._last_row is not None:
            diff_from_last: bool = self._diff(a[0:1], self._last_row)[0]
        else:
            diff_from_last = True
        self._last_row = a[-1:]
        points = np.where(diff)[0].tolist() + [a.shape[0]]
        if len(points) == 1:
            yield (diff_from_last, batch)
        else:
            for i in range(len(points) - 1):
                new_start = diff_from_last if i == 0 else True
                yield (new_start, self.take(batch, points[i], points[i + 1] - points[i]))

    def _diff(self, a: np.ndarray, b: np.ndarray) -> bool:
        return ((a == b) | (a != a) & (b != b)).sum(axis=1) < len(self._keys)