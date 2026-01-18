from __future__ import annotations
import copy
import itertools
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from html import escape
from typing import (
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.treenode import NamedNode, NodePath, Tree
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.datatree_.datatree.common import TreeAttrAccessMixin
from xarray.datatree_.datatree.formatting import datatree_repr
from xarray.datatree_.datatree.formatting_html import (
from xarray.datatree_.datatree.mapping import (
from xarray.datatree_.datatree.ops import (
from xarray.datatree_.datatree.render import RenderTree
def _coerce_to_dataset(data: Dataset | DataArray | None) -> Dataset:
    if isinstance(data, DataArray):
        ds = data.to_dataset()
    elif isinstance(data, Dataset):
        ds = data
    elif data is None:
        ds = Dataset()
    else:
        raise TypeError(f'data object is not an xarray Dataset, DataArray, or None, it is of type {type(data)}')
    return ds