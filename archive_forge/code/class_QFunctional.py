from typing import List
import torch
from torch import Tensor
from torch._ops import ops
class QFunctional(torch.nn.Module):
    """Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> q_add = QFunctional()
        >>> # xdoctest: +SKIP
        >>> a = torch.quantize_per_tensor(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add(a, b)  # Equivalent to ``torch.ops.quantized.add(a, b, 1.0, 0)``

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
        self.scale = 1.0
        self.zero_point = 0
        self.activation_post_process = torch.nn.Identity()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict.pop(prefix + 'scale'))
        self.zero_point = int(state_dict.pop(prefix + 'zero_point'))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def _get_name(self):
        return 'QFunctional'

    def extra_repr(self):
        return f'scale={self.scale}, zero_point={self.zero_point}'

    def forward(self, x):
        raise RuntimeError('Functional is not intended to use the ' + "'forward'. Please use the underlying operation")
    'Operation equivalent to ``torch.ops.quantized.add``'

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.add(Tensor, float)``'

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.add_scalar(x, y)
        return r
    'Operation equivalent to ``torch.ops.quantized.mul(Tensor, Tensor)``'

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.mul(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.mul(Tensor, float)``'

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = ops.quantized.mul_scalar(x, y)
        return r
    'Operation equivalent to ``torch.ops.quantized.cat``'

    def cat(self, x: List[Tensor], dim: int=0) -> Tensor:
        r = ops.quantized.cat(x, scale=self.scale, zero_point=self.zero_point, dim=dim)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.add_relu``'

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.add_relu(x, y, scale=self.scale, zero_point=self.zero_point)
        r = self.activation_post_process(r)
        return r
    'Operation equivalent to ``torch.ops.quantized.matmul(Tensor, Tensor)``'

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = ops.quantized.matmul(x, y, scale=self.scale, zero_point=self.zero_point)
        return r

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == FloatFunctional, 'QFunctional.from_float expects an instance of FloatFunctional'
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = QFunctional()
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod