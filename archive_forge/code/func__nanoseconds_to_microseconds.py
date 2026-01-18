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
def _nanoseconds_to_microseconds(self, time_in_nanoseconds):
    """A helper function for converting nanoseconds to microseconds."""
    time_in_microseconds = time_in_nanoseconds / 1000
    return time_in_microseconds