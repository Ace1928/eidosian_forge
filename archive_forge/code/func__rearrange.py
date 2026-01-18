imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def _rearrange(da, out_dims, in_dims=None, dim_lengths=None):
    """Wrap `einops.rearrange <https://einops.rocks/api/rearrange/>`_.

    This is the function that actually interfaces with ``einops``.
    :func:`xarray_einstats.einops.rearrange` is the user facing
    version as it exposes two possible APIs, one of them significantly
    less verbose and more friendly (but much less flexible).

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be rearranged
    out_dims : list of str, list or dict
        See docstring of :func:`~xarray_einstats.einops.rearrange`
    in_dims : list of str or dict, optional
        See docstring of :func:`~xarray_einstats.einops.rearrange`
    dim_lengths : dict, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`
    """
    if dim_lengths is None:
        dim_lengths = {}
    da_dims = da.dims
    handler = DimHandler()
    if in_dims is None:
        in_dims = []
        in_names = []
        in_pattern = ''
    else:
        in_dims, in_names, in_pattern = process_pattern_list(in_dims, handler=handler, allow_list=False)
    out_dims, out_names, out_pattern = process_pattern_list(out_dims, handler=handler)
    missing_in_dims = [dim for dim in da_dims if dim not in in_names]
    expected_missing = set(out_dims).union(in_names).difference(in_dims)
    missing_out_dims = [dim for dim in da_dims if dim not in expected_missing]
    non_core_dims = [dim for dim in missing_in_dims if dim in missing_out_dims]
    missing_in_dims = [dim for dim in missing_in_dims if dim not in non_core_dims]
    missing_out_dims = [dim for dim in missing_out_dims if dim not in non_core_dims]
    non_core_pattern = handler.get_names(non_core_dims)
    pattern = f'{non_core_pattern} {handler.get_names(missing_in_dims)} {in_pattern} ->        {non_core_pattern} {handler.get_names(missing_out_dims)} {out_pattern}'
    axes_lengths = {handler.rename_kwarg(k): v for k, v in dim_lengths.items() if k in out_names + out_dims}
    kwargs = {k: v for k, v in dim_lengths.items() if k not in out_names + out_dims}
    return xr.apply_ufunc(einops.rearrange, da, pattern, input_core_dims=[missing_in_dims + in_names, []], output_core_dims=[missing_out_dims + out_names], kwargs=axes_lengths, **kwargs)