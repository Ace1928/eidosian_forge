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
class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        shard_embedding_size = config.hidden_size // config.num_hash_functions
        for i in range(config.num_hash_functions):
            name = f'HashBucketCodepointEmbedder_{i}'
            setattr(self, name, nn.Embedding(config.num_hash_buckets, shard_embedding_size))
        self.char_position_embeddings = nn.Embedding(config.num_hash_buckets, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

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

    def forward(self, input_ids: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(input_ids, self.config.hidden_size, self.config.num_hash_functions, self.config.num_hash_buckets)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings