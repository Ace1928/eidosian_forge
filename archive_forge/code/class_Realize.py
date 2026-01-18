import torch.library
from torch import Tensor
from torch.autograd import Function
class Realize(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.ops._inductor_test.realize(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output