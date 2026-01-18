import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
class RandomColorJitter(HybridBlock):
    """Randomly jitters the brightness, contrast, saturation, and hue
    of an image.

    Parameters
    ----------
    brightness : float
        How much to jitter brightness. brightness factor is randomly
        chosen from `[max(0, 1 - brightness), 1 + brightness]`.
    contrast : float
        How much to jitter contrast. contrast factor is randomly
        chosen from `[max(0, 1 - contrast), 1 + contrast]`.
    saturation : float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.
    hue : float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__()
        self._args = (brightness, contrast, saturation, hue)

    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx
        return F.image.random_color_jitter(x, *self._args)