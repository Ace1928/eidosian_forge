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
def _master_process(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool):
    gLogger.info('Running the master process...')
    dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
    remote_em_rref = rpc.remote(self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE))
    remote_net_rref = rpc.remote(self.remote_worker_name(), RemoteNet, args=(D_DENSE + D_SPARSE, D_HID))
    gLogger.info('Created remote rrefs on master')
    self.do_test_on_master(ddp_mode, simulate_uneven_inputs, remote_em_rref, remote_net_rref)