import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
class TensorPipeCudaDistAutogradTest(RpcAgentTestFixture):

    @skip_if_lt_x_gpu(4)
    def test_device_maps_backward_pass(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        t1 = torch.rand(10, device=self.rank, requires_grad=True)
        t2 = torch.rand(10, device=self.rank, requires_grad=True)
        with dist_autograd.context() as context_id:
            res = rpc.rpc_sync(dst, torch.add, args=(t1, t2))
            dist_autograd.backward(context_id, [res.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(torch.ones(10), grads[t1])
            self.assertEqual(torch.ones(10), grads[t2])
            self.assertEqual(t1.device, grads[t1].device)
            self.assertEqual(t2.device, grads[t2].device)
        rpc.shutdown()

    class MyRemoteCompute(torch.nn.Module):

        def forward(self, input):
            input = input * 2.0
            return input

    class MyLocalCompute(torch.nn.Module):

        def __init__(self, next_stage):
            super().__init__()
            self.next_stage = next_stage

        def forward(self, input):
            return self.next_stage.rpc_sync().forward(input)

    @skip_if_lt_x_gpu(4)
    def test_dist_autograd_sync_streams(self):
        options = self.rpc_backend_options
        dst = worker_name((self.rank + 1) % self.world_size)
        options.set_device_map(dst, {self.rank: (self.rank + 1) % self.world_size})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        remote_compute = rpc.remote(dst, TensorPipeCudaDistAutogradTest.MyRemoteCompute)
        local_compute = TensorPipeCudaDistAutogradTest.MyLocalCompute(remote_compute)
        for _ in range(10):
            input = torch.rand([1000, 10000], device=self.rank, requires_grad=True)
            result = input * 2.0
            r = random.random()
            loss = result.sum() * r
            loss.backward()
            with dist_autograd.context() as context_id:
                result = local_compute(input)
                loss = result.sum() * r
                dist_autograd.backward(context_id, [loss])
                grads = dist_autograd.get_gradients(context_id)
                self.assertEqual(input.grad, grads[input])
        rpc.shutdown()

    @skip_if_lt_x_gpu(4)
    def test_gradients_synchronizations(self):
        options = self.rpc_backend_options
        for peer_rank in range(self.world_size):
            options.set_device_map(worker_name(peer_rank), {self.rank: peer_rank})
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
        if self.rank == 0:
            layers = [nn.Linear(2000, 2000) for _ in range(self.world_size - 1)]
            local_layers = [l.to(0) for l in layers]
            remote_layers = []
            for rank in range(1, self.world_size):
                remote_layers.append(rpc.remote(worker_name(rank), WrapperModule, args=(layers[rank - 1], rank)))
            x = torch.randn(5000, 2000).to(0)
            local_model = nn.Sequential(*local_layers)
            local_model(x).sum().backward()
            with dist_autograd.context() as context_id:
                for remote_layer in remote_layers:
                    x = remote_layer.rpc_sync().forward(x)
                dist_autograd.backward(context_id, [x.sum()])
                futs = []
                for remote_layer in remote_layers:
                    futs.append(remote_layer.rpc_async().gradients(context_id))
                for i in range(len(futs)):
                    local_gradients = [p.grad for p in local_layers[i].parameters()]
                    for g1, g2 in zip(futs[i].wait(), local_gradients):
                        self.assertEqual(g1, g2)
        rpc.shutdown()