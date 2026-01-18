from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
def _set_result(self, result: ObjectRefOrListType, meta: 'MetaList', meta_offset: Union[int, List[int]]):
    """
        Set the execution result.

        Parameters
        ----------
        result : ObjectRefOrListType
        meta : MetaList
        meta_offset : int or list of int
        """
    del self.func, self.args, self.kwargs, self.flat_args, self.flat_kwargs
    self.data = result
    self.meta = meta
    self.meta_offset = meta_offset