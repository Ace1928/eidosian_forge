import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import (
from torch.distributed.pipeline.sync import Pipe
def _run_basic_test(self, backend, checkpoint, find_unused_parameters=False, static_graph=False):
    dist.init_process_group(backend=backend, init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name), world_size=self.world_size, rank=self.rank)
    fc1 = nn.Linear(16, 8, bias=False).cuda(2 * self.rank)

    class MyModule(nn.Module):

        def __init__(self, device):
            super().__init__()
            self.fc2 = nn.Linear(8, 4, bias=False).cuda(device)
            self.fc3 = nn.Linear(4, 2, bias=False).cuda(device)

        def forward(self, inp):
            if find_unused_parameters:
                return self.fc2(inp)
            else:
                return self.fc3(self.fc2(inp))
    layer2 = MyModule(2 * self.rank + 1)
    model = nn.Sequential(fc1, layer2)
    model = Pipe(model, chunks=2, checkpoint=checkpoint)
    model = DistributedDataParallel(model, find_unused_parameters=find_unused_parameters, static_graph=static_graph)
    model_input = torch.rand(16, 16).cuda(2 * self.rank) * (self.rank + 1)
    out = model(model_input).local_value()
    out.sum().backward()
    if find_unused_parameters:
        unused_param_input = torch.rand(16, 16).cuda(2 * self.rank) * (self.rank + 1)
        model(unused_param_input).local_value().sum().backward()
    for _ in range(3):
        model_input = torch.rand(16, 16).cuda(2 * self.rank) * (self.rank + 1)
        out = model(model_input).local_value()
        out.sum().backward()
    output = [torch.empty_like(fc1.weight.grad), torch.empty_like(fc1.weight.grad)]
    dist.all_gather(output, fc1.weight.grad)
    self.assertEqual(output[0], output[1])
    output = [torch.empty_like(layer2.fc2.weight.grad), torch.empty_like(layer2.fc2.weight.grad)]
    dist.all_gather(output, layer2.fc2.weight.grad)
    self.assertEqual(output[0], output[1])
    if not find_unused_parameters:
        output = [torch.empty_like(layer2.fc3.weight.grad), torch.empty_like(layer2.fc3.weight.grad)]
        dist.all_gather(output, layer2.fc3.weight.grad)
        self.assertEqual(output[0], output[1])