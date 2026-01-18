import random
from itertools import islice
import numpy as np
import torch
class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool=True, seed: int=0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle)
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas