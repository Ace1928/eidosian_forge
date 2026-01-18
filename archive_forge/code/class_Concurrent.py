import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm
class Concurrent(Sequential):
    """Lays `Block` s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = Concurrent()
        # use net's name_scope to give children blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """

    def __init__(self, axis=-1, prefix=None, params=None):
        super(Concurrent, self).__init__(prefix=prefix, params=params)
        self.axis = axis

    def forward(self, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        out = nd.concat(*out, dim=self.axis)
        return out