imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
class DimHandler:
    """Handle converting actual dimension names to placeholders for einops."""

    def __init__(self):
        self.mapping = {}

    def get_name(self, dim):
        """Return or generate a placeholder for a dimension name."""
        if dim in self.mapping:
            return self.mapping.get(dim)
        dim_txt = f'd{len(self.mapping)}'
        self.mapping[dim] = dim_txt
        return dim_txt

    def get_names(self, dim_list):
        """Automate calling get_name with an iterable."""
        return ' '.join((self.get_name(dim) for dim in dim_list))

    def rename_kwarg(self, key):
        """Process kwargs for axes_lengths.

        Users use as keys the dimension names they used in the input expressions
        which need to be converted and use the placeholder as key when passed
        to einops functions.
        """
        return self.mapping.get(key, key)