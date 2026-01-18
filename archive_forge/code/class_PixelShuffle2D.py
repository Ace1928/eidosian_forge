import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
class PixelShuffle2D(HybridBlock):
    """Pixel-shuffle layer for upsampling in 2 dimensions.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``H`` and ``W`` dimensions, thereby effectively multiplying
    those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^2 C, H, W)` is reshaped
    into :math:`(C, fH, fW)` by forming little :math:`f \\times f` blocks
    of pixels and arranging them in an :math:`H \\times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 2-tuple of int
        Upsampling factors, applied to the ``H`` and ``W`` dimensions,
        in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*C, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, H*f1, W*f2)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle2D((2, 3))
    >>> x = mx.nd.zeros((1, 12, 3, 5))
    >>> pxshuf(x).shape
    (1, 2, 6, 15)
    """

    def __init__(self, factor):
        super(PixelShuffle2D, self).__init__()
        try:
            self._factors = (int(factor),) * 2
        except TypeError:
            self._factors = tuple((int(fac) for fac in factor))
            assert len(self._factors) == 2, 'wrong length {}'.format(len(self._factors))

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        f1, f2 = self._factors
        x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))
        x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))
        x = F.transpose(x, (0, 1, 4, 2, 5, 3))
        x = F.reshape(x, (0, 0, -3, -3))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._factors)