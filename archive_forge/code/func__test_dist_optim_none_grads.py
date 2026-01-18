import threading
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def _test_dist_optim_none_grads(self, optim_cls, *args, **kwargs):
    module1 = MyModule()
    module2 = MyModule(requires_grad=False)
    params = [module1.get_w(), module2.get_w()]
    local_optim = optim_cls(params, *args, **kwargs)
    old_w1 = module1.w.clone().detach()
    old_w2 = module2.w.clone().detach()
    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)
    t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
    t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
    output1 = module1.forward(t2)
    output2 = module2.forward(output1)
    loss = torch.add(output2, t1).sum()
    loss.backward()
    local_optim.step()
    owner1 = 'worker%d' % ((self.rank + 1) % self.world_size)
    owner2 = 'worker%d' % ((self.rank + 2) % self.world_size)
    remote_module1 = rpc.remote(owner1, MyModule)
    remote_module2 = rpc.remote(owner2, MyModule, args=(False,))
    remote_param1 = remote_module1.remote().get_w()
    remote_param2 = remote_module2.remote().get_w()
    self.assertEqual(old_w1, remote_param1.to_here())
    self.assertEqual(old_w2, remote_param2.to_here())
    dist_optim = DistributedOptimizer(optim_cls, [remote_param1, remote_param2], *args, **kwargs)
    with dist_autograd.context() as context_id:
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = remote_module1.rpc_async().forward(t2)
        output2 = remote_module2.rpc_async().forward(output1.wait())
        loss = torch.add(output2.wait(), t1)
        dist_autograd.backward(context_id, [loss.sum()])
        dist_optim.step(context_id)
        new_w1 = remote_module1.rpc_async().get_w().wait()
        new_w2 = remote_module2.rpc_async().get_w().wait()
        self.assertNotEqual(old_w1, new_w1)
        self.assertEqual(old_w2, new_w2)
        self.assertEqual(new_w1, module1.get_w())
        self.assertEqual(new_w2, module2.get_w())