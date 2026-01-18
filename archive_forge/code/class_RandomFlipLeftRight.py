import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
class RandomFlipLeftRight(HybridBlock):
    """Randomly flip the input image left to right with a probability
    of 0.5.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(self):
        super(RandomFlipLeftRight, self).__init__()

    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx
        return F.image.random_flip_left_right(x)