import os
from ....context import cpu
from ...block import HybridBlock
from ... import nn
from .... import base
from .... util import is_np_array
class BottleneckV2(HybridBlock):
    """Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False, in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        act = F.npx.activation if is_np_array() else F.Activation
        x = act(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = act(x, act_type='relu')
        x = self.conv2(x)
        x = self.bn3(x)
        x = act(x, act_type='relu')
        x = self.conv3(x)
        return x + residual