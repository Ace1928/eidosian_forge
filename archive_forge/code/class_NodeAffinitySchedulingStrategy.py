from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class NodeAffinitySchedulingStrategy:
    """Static scheduling strategy used to run a task or actor on a particular node.

    Attributes:
        node_id: the hex id of the node where the task or actor should run.
        soft: whether the scheduler should run the task or actor somewhere else
            if the target node doesn't exist (e.g. the node dies) or is infeasible
            during scheduling.
            If the node exists and is feasible, the task or actor
            will only be scheduled there.
            This means if the node doesn't have the available resources,
            the task or actor will wait indefinitely until resources become available.
            If the node doesn't exist or is infeasible, the task or actor
            will fail if soft is False
            or be scheduled somewhere else if soft is True.
    """

    def __init__(self, node_id: str, soft: bool, _spill_on_unavailable: bool=False, _fail_on_unavailable: bool=False):
        if not isinstance(node_id, str):
            node_id = node_id.hex()
        self.node_id = node_id
        self.soft = soft
        self._spill_on_unavailable = _spill_on_unavailable
        self._fail_on_unavailable = _fail_on_unavailable