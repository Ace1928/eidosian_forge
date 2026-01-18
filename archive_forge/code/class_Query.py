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
@dataclass
class Query:
    args: List[Any]
    kwargs: Dict[Any, Any]
    metadata: RequestMetadata

    async def replace_known_types_in_args(self):
        """Uses the `_PyObjScanner` to find and replace known types.

        1) Replaces `asyncio.Task` objects with their results. This is used for the old
           serve handle API and should be removed once that API is deprecated & removed.
        2) Replaces `DeploymentResponse` objects with their resolved object refs. This
           enables composition without explicitly calling `_to_object_ref`.
        """
        from ray.serve.handle import DeploymentResponse, DeploymentResponseGenerator, _DeploymentResponseBase
        scanner = _PyObjScanner(source_type=(asyncio.Task, _DeploymentResponseBase))
        try:
            tasks = []
            responses = []
            replacement_table = {}
            objs = scanner.find_nodes((self.args, self.kwargs))
            for obj in objs:
                if isinstance(obj, asyncio.Task):
                    tasks.append(obj)
                elif isinstance(obj, DeploymentResponseGenerator):
                    raise RuntimeError('Streaming deployment handle results cannot be passed to downstream handle calls. If you have a use case requiring this feature, please file a feature request on GitHub.')
                elif isinstance(obj, DeploymentResponse):
                    responses.append(obj)
            for task in tasks:
                if hasattr(task, '_ray_serve_object_ref_future'):
                    future = task._ray_serve_object_ref_future
                    replacement_table[task] = await asyncio.wrap_future(future)
                else:
                    replacement_table[task] = task
            if len(responses) > 0:
                obj_refs = await asyncio.gather(*[r._to_object_ref() for r in responses])
                replacement_table.update(zip(responses, obj_refs))
            self.args, self.kwargs = scanner.replace_nodes(replacement_table)
        finally:
            scanner.clear()