import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
def _try_to_scale_up_cluster(topology: Topology, execution_id: str):
    """Try to scale up the cluster to accomodate the provided in-progress workload.

    This makes a resource request to Ray's autoscaler consisting of the current,
    aggregate usage of all operators in the DAG + the incremental usage of all operators
    that are ready for dispatch (i.e. that have inputs queued). If the autoscaler were
    to grant this resource request, it would allow us to dispatch one task for every
    ready operator.

    Note that this resource request does not take the global resource limits or the
    liveness policy into account; it only tries to make the existing resource usage +
    one more task per ready operator feasible in the cluster.

    Args:
        topology: The execution state of the in-progress workload for which we wish to
            request more resources.
    """
    resource_request = []

    def to_bundle(resource: ExecutionResources) -> Dict:
        req = {}
        if resource.cpu:
            req['CPU'] = math.ceil(resource.cpu)
        if resource.gpu:
            req['GPU'] = math.ceil(resource.gpu)
        return req
    for op, state in topology.items():
        per_task_resource = op.incremental_resource_usage()
        task_bundle = to_bundle(per_task_resource)
        resource_request.extend([task_bundle] * op.num_active_tasks())
        if state.num_queued() > 0:
            resource_request.append(task_bundle)
    actor = get_or_create_autoscaling_requester_actor()
    actor.request_resources.remote(resource_request, execution_id)