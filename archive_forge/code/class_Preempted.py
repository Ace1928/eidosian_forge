from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Type
from simpy.core import BoundClass, Environment, SimTime
from simpy.resources import base
class Preempted:
    """Cause of a preemption :class:`~simpy.exceptions.Interrupt` containing
    information about the preemption.

    """

    def __init__(self, by: Optional[Process], usage_since: Optional[SimTime], resource: Resource):
        self.by = by
        'The preempting :class:`simpy.events.Process`.'
        self.usage_since = usage_since
        'The simulation time at which the preempted process started to use\n        the resource.'
        self.resource = resource
        'The resource which was lost, i.e., caused the preemption.'