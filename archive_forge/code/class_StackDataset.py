import bisect
import warnings
import math
from typing import (
from torch import default_generator, randperm
from torch._utils import _accumulate
from ... import Generator, Tensor
class StackDataset(Dataset[T_stack]):
    """Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """
    datasets: Union[tuple, dict]

    def __init__(self, *args: Dataset[T_co], **kwargs: Dataset[T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError('Supported either ``tuple``- (via ``args``) or``dict``- (via ``kwargs``) like input/output, but both types are given.')
            self._length = len(args[0])
            if any((self._length != len(dataset) for dataset in args)):
                raise ValueError('Size mismatch between datasets')
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])
            if any((self._length != len(dataset) for dataset in tmp)):
                raise ValueError('Size mismatch between datasets')
            self.datasets = kwargs
        else:
            raise ValueError('At least one dataset should be passed')

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple((dataset[index] for dataset in self.datasets))

    def __getitems__(self, indices: list):
        if isinstance(self.datasets, dict):
            dict_batch: List[T_dict] = [{} for _ in indices]
            for k, dataset in self.datasets.items():
                if callable(getattr(dataset, '__getitems__', None)):
                    items = dataset.__getitems__(indices)
                    if len(items) != len(indices):
                        raise ValueError(f"Nested dataset's output size mismatch. Expected {len(indices)}, got {len(items)}")
                    for data, d_sample in zip(items, dict_batch):
                        d_sample[k] = data
                else:
                    for idx, d_sample in zip(indices, dict_batch):
                        d_sample[k] = dataset[idx]
            return dict_batch
        list_batch: List[list] = [[] for _ in indices]
        for dataset in self.datasets:
            if callable(getattr(dataset, '__getitems__', None)):
                items = dataset.__getitems__(indices)
                if len(items) != len(indices):
                    raise ValueError(f"Nested dataset's output size mismatch. Expected {len(indices)}, got {len(items)}")
                for data, t_sample in zip(items, list_batch):
                    t_sample.append(data)
            else:
                for idx, t_sample in zip(indices, list_batch):
                    t_sample.append(dataset[idx])
        tuple_batch: List[T_tuple] = [tuple(sample) for sample in list_batch]
        return tuple_batch

    def __len__(self):
        return self._length