from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
class ZarrBackendEntrypoint(BackendEntrypoint):
    """
    Backend for ".zarr" files based on the zarr package.

    For more information about the underlying library, visit:
    https://zarr.readthedocs.io/en/stable

    See Also
    --------
    backends.ZarrStore
    """
    description = 'Open zarr files (.zarr) using zarr in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.ZarrBackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {'.zarr'}
        return False

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, group=None, mode='r', synchronizer=None, consolidated=None, chunk_store=None, storage_options=None, stacklevel=3, zarr_version=None) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = ZarrStore.open_group(filename_or_obj, group=group, mode=mode, synchronizer=synchronizer, consolidated=consolidated, consolidate_on_close=False, chunk_store=chunk_store, storage_options=storage_options, stacklevel=stacklevel + 1, zarr_version=zarr_version)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds

    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs) -> DataTree:
        import zarr
        from xarray.backends.api import open_dataset
        from xarray.core.datatree import DataTree
        from xarray.core.treenode import NodePath
        zds = zarr.open_group(filename_or_obj, mode='r')
        ds = open_dataset(filename_or_obj, engine='zarr', **kwargs)
        tree_root = DataTree.from_dict({'/': ds})
        for path in _iter_zarr_groups(zds):
            try:
                subgroup_ds = open_dataset(filename_or_obj, engine='zarr', group=path, **kwargs)
            except zarr.errors.PathNotFoundError:
                subgroup_ds = Dataset()
            node_name = NodePath(path).name
            new_node: DataTree = DataTree(name=node_name, data=subgroup_ds)
            tree_root._set_item(path, new_node, allow_overwrite=False, new_nodes_along_path=True)
        return tree_root