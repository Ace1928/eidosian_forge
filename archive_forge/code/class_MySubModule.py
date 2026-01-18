import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond
class MySubModule(torch.nn.Module):

    def foo(self, x):
        return x.cos()

    def forward(self, x):
        return self.foo(x)