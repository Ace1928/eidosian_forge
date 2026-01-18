import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class _User2DTransform(Transform):
    """A transform defined by two user-set functions."""
    input_dims = output_dims = 2

    def __init__(self, forward, backward):
        """
        Parameters
        ----------
        forward, backward : callable
            The forward and backward transforms, taking ``x`` and ``y`` as
            separate arguments and returning ``(tr_x, tr_y)``.
        """
        super().__init__()
        self._forward = forward
        self._backward = backward

    def transform_non_affine(self, values):
        return np.transpose(self._forward(*np.transpose(values)))

    def inverted(self):
        return type(self)(self._backward, self._forward)