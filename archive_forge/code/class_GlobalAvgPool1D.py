from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class GlobalAvgPool1D(_Pooling):
    """Global average pooling operation for temporal data.

    Parameters
    ----------
    layout : str, default 'NCW'
        Dimension ordering of data and out ('NCW' or 'NWC').
        'N', 'C', 'W' stands for batch, channel, and width (time) dimensions
        respectively. padding is applied on 'W' dimension.


    Inputs:
        - **data**: 3D input tensor with shape `(batch_size, in_channels, width)`
          when `layout` is `NCW`. For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 3D output tensor with shape `(batch_size, channels, 1)`.
    """

    def __init__(self, layout='NCW', **kwargs):
        assert layout in ('NCW', 'NWC'), 'Only NCW and NWC layouts are valid for 1D Pooling'
        super(GlobalAvgPool1D, self).__init__((1,), None, 0, True, True, 'avg', layout, **kwargs)