from __future__ import annotations
import functools
class Kernel(BuiltinFilter):
    """
    Create a convolution kernel. The current version only
    supports 3x3 and 5x5 integer and floating point kernels.

    In the current version, kernels can only be applied to
    "L" and "RGB" images.

    :param size: Kernel size, given as (width, height). In the current
                    version, this must be (3,3) or (5,5).
    :param kernel: A sequence containing kernel weights. The kernel will
                   be flipped vertically before being applied to the image.
    :param scale: Scale factor. If given, the result for each pixel is
                    divided by this value. The default is the sum of the
                    kernel weights.
    :param offset: Offset. If given, this value is added to the result,
                    after it has been divided by the scale factor.
    """
    name = 'Kernel'

    def __init__(self, size, kernel, scale=None, offset=0):
        if scale is None:
            scale = functools.reduce(lambda a, b: a + b, kernel)
        if size[0] * size[1] != len(kernel):
            msg = 'not enough coefficients in kernel'
            raise ValueError(msg)
        self.filterargs = (size, scale, offset, kernel)