import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def _generate_nccl_uid(self, key):
    """Generate an NCCL unique ID for initializing communicators.

        The method will also create a KV store using Ray named actor and store
        the NCCLUniqueID in the store. The store needs to be garbage collected
        when destroying the collective group.

        Args:
            key: the key of the .

        Returns:
            NCCLUniqueID (str): NCCL unique ID.
        """
    group_uid = nccl_util.get_nccl_unique_id()
    store_name = get_store_name(key)
    from ray.util.collective.util import NCCLUniqueIDStore
    store = NCCLUniqueIDStore.options(name=store_name, lifetime='detached').remote(store_name)
    ray.get([store.set_id.remote(group_uid)])
    return group_uid