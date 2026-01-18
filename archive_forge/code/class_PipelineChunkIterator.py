import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
class PipelineChunkIterator(PipelineIterator):

    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or any iterator):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
        """
        super().__init__(loader, infer, params)

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            processed = next(self.subiterator)
        except StopIteration:
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed