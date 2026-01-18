from typing import Optional
from tensorflow import nest
from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
class GeneralBlock(block_module.Block):
    """A general neural network block when the input type is unknown.

    When the input type is unknown. The GeneralBlock would search in a large space
    for a good model.

    # Arguments
        name: String.
    """

    def build(self, hp, inputs=None):
        raise NotImplementedError