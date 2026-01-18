from typing import List
import torch
from torch import Tensor
from torch._ops import ops
class FloatFunctional(torch.nn.Module):
    """State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self):
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError('FloatFunctional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.add(Tensor, Tensor)``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        return r
    'Operation equivalent to ``torch.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        return r
    'Operation equivalent to ``torch.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``relu(torch.add(x,y))``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r