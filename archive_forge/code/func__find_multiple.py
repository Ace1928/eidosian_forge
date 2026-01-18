import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
import pyarrow.parquet as pq
import orjson
from ochat.training_deepspeed.multipack_sampler import MultipackDistributedSampler
def _find_multiple(a, b):
    return -(a // -b) * b