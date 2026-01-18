import math
from typing import Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .module import Module
from .lazy import LazyModuleMixin
class LazyLinear(LazyModuleMixin, Linear):
    """A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are
            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where
            :math:`k = \\frac{1}{\\text{in\\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
                :math:`k = \\frac{1}{\\text{in\\_features}}`


    """
    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, out_features: int, bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()