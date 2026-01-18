import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
    """
        Converts ids to hash bucket ids via multiple hashing.

        Args:
            input_ids: The codepoints or other IDs to be hashed.
            num_hashes: The number of hash functions to use.
            num_buckets: The number of hash buckets (i.e. embeddings in each table).

        Returns:
            A list of tensors, each of which is the hash bucket IDs from one hash function.
        """
    if num_hashes > len(_PRIMES):
        raise ValueError(f'`num_hashes` must be <= {len(_PRIMES)}')
    primes = _PRIMES[:num_hashes]
    result_tensors = []
    for prime in primes:
        hashed = (input_ids + 1) * prime % num_buckets
        result_tensors.append(hashed)
    return result_tensors