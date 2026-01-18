from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Type
from simpy.core import BoundClass, Environment, SimTime
from simpy.resources import base
class PriorityResource(Resource):
    """A :class:`~simpy.resources.resource.Resource` supporting prioritized
    requests.

    Pending requests in the :attr:`~Resource.queue` are sorted in ascending
    order by their *priority* (that means lower values are more important).

    """
    PutQueue = SortedQueue
    'Type of the put queue. See\n    :attr:`~simpy.resources.base.BaseResource.put_queue` for details.'
    GetQueue = list
    'Type of the get queue. See\n    :attr:`~simpy.resources.base.BaseResource.get_queue` for details.'

    def __init__(self, env: Environment, capacity: int=1):
        super().__init__(env, capacity)
    if TYPE_CHECKING:

        def request(self, priority: int=0, preempt: bool=True) -> PriorityRequest:
            """Request a usage slot with the given *priority*."""
            return PriorityRequest(self, priority, preempt)

        def release(self, request: PriorityRequest) -> Release:
            """Release a usage slot."""
            return Release(self, request)
    else:
        request = BoundClass(PriorityRequest)
        release = BoundClass(Release)