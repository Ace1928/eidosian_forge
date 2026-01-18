imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
class DaskBackend(einops._backends.AbstractBackend):
    """Dask backend class for einops.

    It should be imported before using functions of :mod:`xarray_einstats.einops`
    on Dask backed DataArrays.
    It doesn't need to be initialized or used explicitly

    Notes
    -----
    Class created from the advise on
    `issue einops#120 <https://github.com/arogozhnikov/einops/issues/120>`_ about Dask support.
    And from reading
    `einops/_backends <https://github.com/arogozhnikov/einops/blob/master/einops/_backends.py>`_,
    the source of the AbstractBackend class of which DaskBackend is a subclass.
    """
    framework_name = 'dask'

    def __init__(self):
        """Initialize DaskBackend.

        Contains the imports to avoid errors when dask is not installed
        """
        import dask.array as dsar
        self.dsar = dsar

    def is_appropriate_type(self, tensor):
        """Recognizes tensors it can handle."""
        return isinstance(tensor, self.dsar.core.Array)

    def from_numpy(self, x):
        return self.dsar.array(x)

    def to_numpy(self, x):
        return x.compute()

    def arange(self, start, stop):
        return self.dsar.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.dsar.stack(tensors)

    def tile(self, x, repeats):
        return self.dsar.tile(x, repeats)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128')

    def add_axis(self, x, new_position):
        return self.dsar.expand_dims(x, new_position)