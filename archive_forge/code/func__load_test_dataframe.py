from __future__ import annotations
from types import ModuleType
from typing import Type
import numpy as np
import pandas
import pytest
import ray
import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler
import modin.pandas as pd
from modin.experimental.torch.datasets import ModinDataLoader
def _load_test_dataframe(lib: ModuleType):
    df = lib.read_csv('https://raw.githubusercontent.com/ponder-org/ponder-datasets/main/USA_Housing.csv')
    return df