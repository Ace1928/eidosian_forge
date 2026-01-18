import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
class PipelineIterator(IterableDataset):

    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for item in loader:
            yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or any iterator):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
                    loader_batch_size (`int`, *optional*):
                        If specified, the items of `loader` are supposed to come as batch, and are loader_batched here
                        making it roughly behave as


        ```
        for items in loader:
            for i in loader_batch_size:
                item = items[i]
                yield infer(item, **params)
        ```"""
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size
        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        if isinstance(self._loader_batch_data, torch.Tensor):
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if isinstance(element, ModelOutput):
                    element = element.to_tuple()
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                    continue
                if k in {'hidden_states', 'past_key_values', 'attentions'} and isinstance(element, tuple):
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple((el[self._loader_batch_index].unsqueeze(0) for el in element))
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple((np.expand_dims(el[self._loader_batch_index], 0) for el in element))
                    continue
                if element is None:
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    loader_batched[k] = element[self._loader_batch_index]
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            return self.loader_batch_item()
        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        if self.loader_batch_size is not None:
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                self.loader_batch_size = observed_batch_size
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            return processed