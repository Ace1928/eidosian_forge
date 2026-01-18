import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
class GPT2(Base):
    """
    GPT2 pytorch implementation, for testing purposes in the image-GPT context
    Credits: https://github.com/teddykoker/image-gpt"""

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, num_positions: int, num_vocab: int, num_classes: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)
        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_Block(embed_dim, num_heads))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor, classify: bool=False) -> Any:
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape
        h = self.token_embeddings(x)
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], dim=0)
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h)
        if not classify:
            return logits
        h = torch.mean(h, dim=0)
        return (self.clf_head(h), logits)