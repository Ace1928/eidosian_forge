import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class RemoteEM(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        gLogger.info('Initing RemoteEM with %s %s', num_embeddings, embedding_dim)
        super().__init__()
        init_em = [0.5] * embedding_dim
        self.em = nn.EmbeddingBag(num_embeddings, embedding_dim, _weight=torch.tensor([init_em] * num_embeddings))

    def forward(self, input: torch.Tensor):
        gLogger.debug('Running RemoteEM.forward() on: %s', input)
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))