import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
class PixelShuffle1D(HybridBlock):
    """Pixel-shuffle layer for upsampling in 1 dimension.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``W`` dimension, thereby effectively multiplying that dimension
    by a constant factor in size.

    For example, a feature map of shape :math:`(fC, W)` is reshaped
    into :math:`(C, fW)` by forming little value groups of size :math:`f`
    and arranging them in a grid of size :math:`W`.

    Parameters
    ----------
    factor : int or 1-tuple of int
        Upsampling factor, applied to the ``W`` dimension.

    Inputs:
        - **data**: Tensor of shape ``(N, f*C, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, W*f)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle1D(2)
    >>> x = mx.nd.zeros((1, 8, 3))
    >>> pxshuf(x).shape
    (1, 4, 6)
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self._factor = int(factor)

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        f = self._factor
        x = F.reshape(x, (0, -4, -1, f, 0))
        x = F.transpose(x, (0, 1, 3, 2))
        x = F.reshape(x, (0, 0, -3))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._factor)