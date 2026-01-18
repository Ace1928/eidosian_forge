from typing import List
import torch
from torch import Tensor
from torch._ops import ops
class FXFloatFunctional(torch.nn.Module):
    """ module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def forward(self, x):
        raise RuntimeError('FloatFunctional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.add(Tensor, Tensor)``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        r = torch.cat(x, dim=dim)
        return r
    'Operation equivalent to ``relu(torch.add(x,y))``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        return r
    'Operation equivalent to ``torch.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        return r