import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from ..utils.generic import ModelOutput
class KeyDataset(Dataset):

    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]