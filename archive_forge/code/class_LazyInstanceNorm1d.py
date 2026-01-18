import warnings
from torch import Tensor
from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F
class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
    """A :class:`torch.nn.InstanceNorm1d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`, `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`(C, L)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)
    """
    cls_to_become = InstanceNorm1d

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f'expected 2D or 3D input (got {input.dim()}D input)')