from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
class _Pooling(HybridBlock):
    """Abstract class for different pooling layers."""

    def __init__(self, pool_size, strides, padding, ceil_mode, global_pool, pool_type, layout, count_include_pad=None, **kwargs):
        super(_Pooling, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        if isinstance(strides, numeric_types):
            strides = (strides,) * len(pool_size)
        if isinstance(padding, numeric_types):
            padding = (padding,) * len(pool_size)
        self._kwargs = {'kernel': pool_size, 'stride': strides, 'pad': padding, 'global_pool': global_pool, 'pool_type': pool_type, 'layout': layout, 'pooling_convention': 'full' if ceil_mode else 'valid'}
        if count_include_pad is not None:
            self._kwargs['count_include_pad'] = count_include_pad

    def _alias(self):
        return 'pool'

    def hybrid_forward(self, F, x):
        pooling = F.npx.pooling if is_np_array() else F.Pooling
        return pooling(x, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}(size={kernel}, stride={stride}, padding={pad}, ceil_mode={ceil_mode}'
        s += ', global_pool={global_pool}, pool_type={pool_type}, layout={layout})'
        return s.format(name=self.__class__.__name__, ceil_mode=self._kwargs['pooling_convention'] == 'full', **self._kwargs)