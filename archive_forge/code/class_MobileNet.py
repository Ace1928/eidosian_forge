import os
from ... import nn
from ....context import cpu
from ...block import HybridBlock
from .... import base
class MobileNet(HybridBlock):
    """MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                _add_conv(self.features, channels=int(32 * multiplier), kernel=3, pad=1, stride=2)
                dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024]]
                channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2]
                strides = [1, 2] * 3 + [1] * 5 + [2, 1]
                for dwc, c, s in zip(dw_channels, channels, strides):
                    _add_conv_dw(self.features, dw_channels=dwc, channels=c, stride=s)
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x