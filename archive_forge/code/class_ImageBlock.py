from typing import Optional
from tensorflow import nest
from autokeras.blocks import basic
from autokeras.blocks import preprocessing
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
class ImageBlock(block_module.Block):
    """Block for image data.

    The image blocks is a block choosing from ResNetBlock, XceptionBlock, ConvBlock,
    which is controlled by a hyperparameter, 'block_type'.

    # Arguments
        block_type: String. 'resnet', 'xception', 'vanilla'. The type of Block
            to use. If unspecified, it will be tuned automatically.
        normalize: Boolean. Whether to channel-wise normalize the images.
            If unspecified, it will be tuned automatically.
        augment: Boolean. Whether to do image augmentation. If unspecified,
            it will be tuned automatically.
    """

    def __init__(self, block_type: Optional[str]=None, normalize: Optional[bool]=None, augment: Optional[bool]=None, **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.normalize = normalize
        self.augment = augment

    def get_config(self):
        config = super().get_config()
        config.update({BLOCK_TYPE: self.block_type, NORMALIZE: self.normalize, AUGMENT: self.augment})
        return config

    def _build_block(self, hp, output_node, block_type):
        if block_type == RESNET:
            return basic.ResNetBlock().build(hp, output_node)
        elif block_type == XCEPTION:
            return basic.XceptionBlock().build(hp, output_node)
        elif block_type == VANILLA:
            return basic.ConvBlock().build(hp, output_node)
        elif block_type == EFFICIENT:
            return basic.EfficientNetBlock().build(hp, output_node)

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        output_node = input_node
        if self.normalize is None and hp.Boolean(NORMALIZE):
            with hp.conditional_scope(NORMALIZE, [True]):
                output_node = preprocessing.Normalization().build(hp, output_node)
        elif self.normalize:
            output_node = preprocessing.Normalization().build(hp, output_node)
        if self.augment is None and hp.Boolean(AUGMENT):
            with hp.conditional_scope(AUGMENT, [True]):
                output_node = preprocessing.ImageAugmentation().build(hp, output_node)
        elif self.augment:
            output_node = preprocessing.ImageAugmentation().build(hp, output_node)
        if self.block_type is None:
            block_type = hp.Choice(BLOCK_TYPE, [RESNET, XCEPTION, VANILLA, EFFICIENT])
            with hp.conditional_scope(BLOCK_TYPE, [block_type]):
                output_node = self._build_block(hp, output_node, block_type)
        else:
            output_node = self._build_block(hp, output_node, self.block_type)
        return output_node