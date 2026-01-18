import asyncio
import enum
import logging
import math
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
import ray
from ray._private.utils import load_class
from ray.actor import ActorHandle
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.exceptions import RayActorError
from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.utils import JavaActorHandleProxy, MetricsPusher
from ray.serve.generated.serve_pb2 import DeploymentRoute
from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
from ray.serve.grpc_util import RayServegRPCContext
from ray.util import metrics
def _get_candidate_multiplexed_replica_ids(self, model_id: str, get_from_all_replicas: bool=False) -> Set[str]:
    """Get multiplexed model candidates from the current replica.

        By default, we will only choose from replicas that have the requested
        multiplexed model id, if not matched, the function will return an empty set.

        If get_from_all_replicas is True, we will choose from all replicas,
        and we will choose all replicas with the least number of multiplexed model
        ids.

        """
    candidates = set()
    if not get_from_all_replicas:
        if model_id in self._multiplexed_model_id_to_replica_ids:
            candidates = self._multiplexed_model_id_to_replica_ids[model_id]
            if len(candidates) > 0:
                return candidates
        return candidates
    sorted_replicas = sorted(self._replicas.values(), key=lambda x: len(x.multiplexed_model_ids))
    least_num_multiplexed_model_ids = math.inf
    for replica in sorted_replicas:
        if len(replica.multiplexed_model_ids) <= least_num_multiplexed_model_ids:
            candidates.add(replica.replica_id)
            least_num_multiplexed_model_ids = len(replica.multiplexed_model_ids)
        else:
            break
    return candidates