import dataclasses
import inspect
import logging
from collections import defaultdict
from functools import wraps
from typing import List, Optional, Tuple
import aiohttp
import grpc
from grpc.aio._call import UnaryStreamCall
import ray
import ray.dashboard.modules.log.log_consts as log_consts
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import hex_to_binary
from ray._raylet import ActorID, JobID, TaskID
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated.gcs_pb2 import ActorTableData
from ray.core.generated.gcs_service_pb2 import (
from ray.core.generated.node_manager_pb2 import (
from ray.core.generated.node_manager_pb2_grpc import NodeManagerServiceStub
from ray.core.generated.reporter_pb2 import (
from ray.core.generated.reporter_pb2_grpc import LogServiceStub
from ray.core.generated.runtime_env_agent_pb2 import (
from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.job.common import JobInfoStorageClient
from ray.dashboard.modules.job.pydantic_models import JobDetails, JobType
from ray.dashboard.modules.job.utils import get_driver_jobs
from ray.dashboard.utils import Dict as Dictionary
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
class IdToIpMap:

    def __init__(self):
        self._ip_to_node_id = defaultdict(str)
        self._node_id_to_ip = defaultdict(str)

    def put(self, node_id: str, address: str):
        self._ip_to_node_id[address] = node_id
        self._node_id_to_ip[node_id] = address

    def get_ip(self, node_id: str):
        return self._node_id_to_ip.get(node_id)

    def get_node_id(self, address: str):
        return self._ip_to_node_id.get(address)

    def pop(self, node_id: str):
        """Pop the given node id.

        Returns:
            False if the corresponding node id doesn't exist.
            True if it pops correctly.
        """
        ip = self._node_id_to_ip.get(node_id)
        if not ip:
            return None
        assert ip in self._ip_to_node_id
        self._node_id_to_ip.pop(node_id)
        self._ip_to_node_id.pop(ip)
        return True