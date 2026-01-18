import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def actor_table(self, actor_id: str, job_id: ray.JobID=None, actor_state_name: str=None):
    """Fetch and parse the actor table information for a single actor ID.

        Args:
            actor_id: A hex string of the actor ID to fetch information about.
                If this is None, then the actor table is fetched.
                If this is not None, `job_id` and `actor_state_name`
                will not take effect.
            job_id: To filter actors by job_id, which is of type `ray.JobID`.
                You can use the `ray.get_runtime_context().job_id` function
                to get the current job ID
            actor_state_name: To filter actors based on actor state,
                which can be one of the following: "DEPENDENCIES_UNREADY",
                "PENDING_CREATION", "ALIVE", "RESTARTING", or "DEAD".
        Returns:
            Information from the actor table.
        """
    self._check_connected()
    if actor_id is not None:
        actor_id = ray.ActorID(hex_to_binary(actor_id))
        actor_info = self.global_state_accessor.get_actor_info(actor_id)
        if actor_info is None:
            return {}
        else:
            actor_table_data = gcs_pb2.ActorTableData.FromString(actor_info)
            return self._gen_actor_info(actor_table_data)
    else:
        validate_actor_state_name(actor_state_name)
        actor_table = self.global_state_accessor.get_actor_table(job_id, actor_state_name)
        results = {}
        for i in range(len(actor_table)):
            actor_table_data = gcs_pb2.ActorTableData.FromString(actor_table[i])
            results[binary_to_hex(actor_table_data.actor_id)] = self._gen_actor_info(actor_table_data)
        return results