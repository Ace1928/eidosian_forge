from __future__ import annotations
import abc
import copy
import html
from collections.abc import (
from typing import Any
import tlz as toolz
import dask
from dask import config
from dask.base import clone_key, flatten, is_dask_collection, normalize_token
from dask.core import keys_in_tasks, reverse_dict
from dask.typing import DaskCollection, Graph, Key
from dask.utils import ensure_dict, import_required, key_split
from dask.widgets import get_template
def _toposort_layers(self) -> list[str]:
    """Sort the layers in a high level graph topologically

        Parameters
        ----------
        hlg : HighLevelGraph
            The high level graph's layers to sort

        Returns
        -------
        sorted: list
            List of layer names sorted topologically
        """
    degree = {k: len(v) for k, v in self.dependencies.items()}
    reverse_deps: dict[str, list[str]] = {k: [] for k in self.dependencies}
    ready = []
    for k, v in self.dependencies.items():
        for dep in v:
            reverse_deps[dep].append(k)
        if not v:
            ready.append(k)
    ret = []
    while len(ready) > 0:
        layer = ready.pop()
        ret.append(layer)
        for rdep in reverse_deps[layer]:
            degree[rdep] -= 1
            if degree[rdep] == 0:
                ready.append(rdep)
    return ret