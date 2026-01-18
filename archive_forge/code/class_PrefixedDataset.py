import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
class PrefixedDataset(Mapping):
    """
    Will access keys in a given dataset by adding a prefix.

    Args:
        dataset (`Mapping`): Any map with string keys.
        prefix (`str`): A prefix to add when trying to access any element in the underlying dataset.
    """

    def __init__(self, dataset: Mapping, prefix: str):
        self.dataset = dataset
        self.prefix = prefix

    def __getitem__(self, key):
        return self.dataset[f'{self.prefix}{key}']

    def __iter__(self):
        return iter([key for key in self.dataset if key.startswith(self.prefix)])

    def __len__(self):
        return len(self.dataset)