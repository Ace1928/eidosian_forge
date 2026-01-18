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
class CudaDdpComparisonTest(CommonDdpComparisonTest):

    @skip_if_lt_x_gpu(NUM_TRAINERS)
    @requires_nccl()
    @dist_init
    @skip_if_rocm
    def test_ddp_dist_autograd_local_vs_remote_gpu(self):
        torch.manual_seed(self.rank)
        dist.init_process_group(backend='gloo', init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
        remote_layer1 = RemoteModule(remote_device='worker0/cpu', module_cls=nn.Linear, args=(10, 7, False))
        layer1 = nn.Linear(10, 7, False)
        layer1.weight = remote_layer1.module_rref.to_here().weight
        layer2 = nn.Linear(7, 5).cuda(self.rank)
        ddp_layer2 = DistributedDataParallel(layer2, device_ids=[self.rank])
        remote_layer3 = RemoteModule(remote_device='worker0/cpu', module_cls=nn.Linear, args=(5, 3, False))
        layer3 = nn.Linear(5, 3, False)
        layer3.weight = remote_layer3.module_rref.to_here().weight
        layer4 = nn.Linear(3, 1).cuda(self.rank)
        ddp_layer4 = DistributedDataParallel(layer4, device_ids=[self.rank])
        inputs = torch.rand((10, 10))
        loss = ddp_layer4(layer3(ddp_layer2(layer1(inputs).cuda(self.rank)).cpu()).cuda(self.rank)).sum()
        loss.backward()
        with dist_autograd.context() as context_id:
            loss = ddp_layer4(remote_layer3(ddp_layer2(remote_layer1(inputs).cuda(self.rank)).cpu()).cuda(self.rank)).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            dist.barrier()
            self.assertEqual(layer1.weight.grad, rpc.rpc_sync('worker0', CommonDdpComparisonTest.get_remote_grads, args=(remote_layer1.module_rref, context_id)))
            self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])
            self.assertEqual(layer3.weight.grad, rpc.rpc_sync('worker0', CommonDdpComparisonTest.get_remote_grads, args=(remote_layer3.module_rref, context_id)))
            self.assertEqual(layer4.weight.grad, grads_dict[layer4.weight])