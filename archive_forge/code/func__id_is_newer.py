import inspect
import logging
import os
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._raylet as raylet
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.signature import extract_signature, get_signature
from ray._private.utils import check_oversized_function
from ray.util.client import ray
from ray.util.client.options import validate_options
def _id_is_newer(id1: int, id2: int) -> bool:
    """
    We should only replace cache entries with the responses for newer IDs.
    Most of the time newer IDs will be the ones with higher value, except when
    the req_id counter rolls over. We check for this case by checking the
    distance between the two IDs. If the distance is significant, then it's
    likely that the req_id counter rolled over, and the smaller id should
    still be used to replace the one in cache.
    """
    diff = abs(id2 - id1)
    if diff > INT32_MAX // 2:
        return id1 < id2
    return id1 > id2