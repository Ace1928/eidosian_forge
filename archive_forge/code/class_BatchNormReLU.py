import warnings
import numpy as np
from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent
from ... import nd, sym
from ...util import is_np_array
class BatchNormReLU(_BatchNorm):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    running_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the running mean.
    running_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the running variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, axis=1, momentum=0.9, epsilon=1e-05, center=True, scale=True, use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones', running_mean_initializer='zeros', running_variance_initializer='ones', in_channels=0, **kwargs):
        super(BatchNormReLU, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale, use_global_stats=use_global_stats, fuse_relu=True, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, running_mean_initializer=running_mean_initializer, running_variance_initializer=running_variance_initializer, in_channels=in_channels, **kwargs)