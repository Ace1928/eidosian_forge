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
def _check_connected(self):
    """Ensure that the object has been initialized before it is used.

        This lazily initializes clients needed for state accessors.

        Raises:
            RuntimeError: An exception is raised if ray.init() has not been
                called yet.
        """
    if self.gcs_options is not None and self.global_state_accessor is None:
        self._really_init_global_state()
    if self.global_state_accessor is None:
        raise ray.exceptions.RaySystemError("Ray has not been started yet. You can start Ray with 'ray.init()'.")