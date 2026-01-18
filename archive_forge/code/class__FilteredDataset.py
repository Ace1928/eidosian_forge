import os
from ... import recordio, ndarray
class _FilteredDataset(Dataset):
    """Dataset with a filter applied"""

    def __init__(self, dataset, fn):
        self._dataset = dataset
        self._indices = [i for i, sample in enumerate(dataset) if fn(sample)]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]