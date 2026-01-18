from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def _open_datatree_netcdf(ncDataset: ncDataset | ncDatasetLegacyH5, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs) -> DataTree:
    from xarray.backends.api import open_dataset
    from xarray.core.datatree import DataTree
    from xarray.core.treenode import NodePath
    ds = open_dataset(filename_or_obj, **kwargs)
    tree_root = DataTree.from_dict({'/': ds})
    with ncDataset(filename_or_obj, mode='r') as ncds:
        for path in _iter_nc_groups(ncds):
            subgroup_ds = open_dataset(filename_or_obj, group=path, **kwargs)
            node_name = NodePath(path).name
            new_node: DataTree = DataTree(name=node_name, data=subgroup_ds)
            tree_root._set_item(path, new_node, allow_overwrite=False, new_nodes_along_path=True)
    return tree_root