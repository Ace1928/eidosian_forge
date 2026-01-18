from collections import defaultdict
from ray.util.client.server.server_pickler import loads_from_client
import ray
import logging
import grpc
from queue import Queue
import sys
from typing import Any, Dict, Iterator, TYPE_CHECKING, Union
from threading import Event, Lock, Thread
import time
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.client import CURRENT_PROTOCOL_VERSION
from ray.util.debug import log_once
from ray._private.client_mode_hook import disable_client_hook
def _get_reconnecting_from_context(context: Any) -> bool:
    """
    Get `reconnecting` from gRPC metadata, or False if missing.
    """
    metadata = {k: v for k, v in context.invocation_metadata()}
    val = metadata.get('reconnecting')
    if val is None or val not in ('True', 'False'):
        logger.error(f'Client connecting with invalid value for "reconnecting": {val}, This may be because you have a mismatched client and server version.')
        return False
    return val == 'True'