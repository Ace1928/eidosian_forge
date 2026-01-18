from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
class MetaListHook(MaterializationHook):
    """
    Used by MetaList.__getitem__() for lazy materialization and getting a single value from the list.

    Parameters
    ----------
    meta : MetaList
        Non-materialized list to get the value from.
    idx : int
        The value index in the list.
    """

    def __init__(self, meta: MetaList, idx: int):
        self.meta = meta
        self.idx = idx

    def pre_materialize(self):
        """
        Get item at self.idx or object ref if not materialized.

        Returns
        -------
        object
        """
        obj = self.meta._obj
        return obj[self.idx] if isinstance(obj, list) else obj

    def post_materialize(self, materialized):
        """
        Save the materialized list in self.meta and get the item at self.idx.

        Parameters
        ----------
        materialized : list

        Returns
        -------
        object
        """
        self.meta._obj = materialized
        return materialized[self.idx]