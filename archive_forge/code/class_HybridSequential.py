import warnings
import numpy as np
from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent
from ... import nd, sym
from ...util import is_np_array
class HybridSequential(HybridBlock):
    """Stacks HybridBlocks sequentially.

    Example::

        net = nn.HybridSequential()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
        net.hybridize()
    """

    def __init__(self, prefix=None, params=None):
        super(HybridSequential, self).__init__(prefix=prefix, params=params)

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    def hybrid_forward(self, F, x):
        for block in self._children.values():
            x = block(x)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key, block=_indent(block.__repr__(), 2)) for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)(prefix=self._prefix)
            with net.name_scope():
                net.add(*layers)
            return net
        else:
            return layers

    def __len__(self):
        return len(self._children)