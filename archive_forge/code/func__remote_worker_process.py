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
def _remote_worker_process(self, ddp_mode):
    gLogger.info('The remote worker is running.')
    dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
    if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
        dist.new_group(TRAINER_RANKS)
    global shutdown_signal
    with shutdown_signal:
        shutdown_signal.wait()
    gLogger.info('Exiting remote worker.')
    dist.destroy_process_group()