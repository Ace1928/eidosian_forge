from typing import Optional
from tensorflow import nest
from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
class TimeseriesBlock(block_module.Block):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        output_node = basic.RNNBlock().build(hp, output_node)
        return output_node