import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
def _put_object(self, data: Union[bytes, bytearray], client_ref_id: bytes, client_id: str, owner_id: bytes, context=None):
    """Put an object in the cluster with ray.put() via gRPC.

        Args:
            data: Pickled data. Can either be bytearray if this is called
              from the dataservicer, or bytes if called from PutObject.
            client_ref_id: The id associated with this object on the client.
            client_id: The client who owns this data, for tracking when to
              delete this reference.
            owner_id: The owner id of the object.
            context: gRPC context.
        """
    try:
        obj = loads_from_client(data, self)
        if owner_id:
            owner = self.actor_refs[owner_id]
        else:
            owner = None
        with disable_client_hook():
            objectref = ray.put(obj, _owner=owner)
    except Exception as e:
        logger.exception('Put failed:')
        return ray_client_pb2.PutResponse(id=b'', valid=False, error=cloudpickle.dumps(e))
    self.object_refs[client_id][objectref.binary()] = objectref
    if len(client_ref_id) > 0:
        self.client_side_ref_map[client_id][client_ref_id] = objectref.binary()
    logger.debug('put: %s' % objectref)
    return ray_client_pb2.PutResponse(id=objectref.binary(), valid=True)