from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class GlobalMaxPool3D(_Pooling):
    """Global max pooling operation for 3D data (spatial or spatio-temporal).


    Parameters
    ----------
    layout : str, default 'NCDHW'
        Dimension ordering of data and out ('NCDHW' or 'NDHWC').
        'N', 'C', 'H', 'W', 'D' stands for batch, channel, height, width and
        depth dimensions respectively. padding is applied on 'D', 'H' and 'W'
        dimension.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, in_channels, depth, height, width)` when `layout` is `NCW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, channels, 1, 1, 1)` when `layout` is `NCDHW`.
    """

    def __init__(self, layout='NCDHW', **kwargs):
        assert layout in ('NCDHW', 'NDHWC'), 'Only NCDHW and NDHWC layouts are valid for 3D Pooling'
        super(GlobalMaxPool3D, self).__init__((1, 1, 1), None, 0, True, True, 'max', layout, **kwargs)