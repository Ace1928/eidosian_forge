from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class ReflectionPad2D(HybridBlock):
    """Pads the input tensor using the reflection of the input boundary.

    Parameters
    ----------
    padding: int
        An integer padding size


    Inputs:
        - **data**: input tensor with the shape :math:`(N, C, H_{in}, W_{in})`.

    Outputs:
        - **out**: output tensor with the shape :math:`(N, C, H_{out}, W_{out})`, where

          .. math::

            H_{out} = H_{in} + 2 \\cdot padding

            W_{out} = W_{in} + 2 \\cdot padding


    Examples
    --------
    >>> m = nn.ReflectionPad2D(3)
    >>> input = mx.nd.random.normal(shape=(16, 3, 224, 224))
    >>> output = m(input)
    """

    def __init__(self, padding=0, **kwargs):
        super(ReflectionPad2D, self).__init__(**kwargs)
        if isinstance(padding, numeric_types):
            padding = (0, 0, 0, 0, padding, padding, padding, padding)
        assert len(padding) == 8
        self._padding = padding

    def hybrid_forward(self, F, x):
        return F.pad(x, mode='reflect', pad_width=self._padding)