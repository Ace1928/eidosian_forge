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
def _embed_hash_buckets(self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int):
    """Converts IDs (e.g. codepoints) into embeddings via multiple hashing."""
    if embedding_size % num_hashes != 0:
        raise ValueError(f'Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0')
    hash_bucket_tensors = self._hash_bucket_tensors(input_ids, num_hashes=num_hashes, num_buckets=num_buckets)
    embedding_shards = []
    for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
        name = f'HashBucketCodepointEmbedder_{i}'
        shard_embeddings = getattr(self, name)(hash_bucket_ids)
        embedding_shards.append(shard_embeddings)
    return torch.cat(embedding_shards, dim=-1)