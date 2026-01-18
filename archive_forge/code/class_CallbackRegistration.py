import dataclasses
import datetime
import enum
import math
from typing import Dict, List, Mapping, Optional, Set, Union
from ortools.math_opt import callback_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class CallbackRegistration:
    """Request the events and input data and reports output types for a callback.

    Note that it is an error to add a constraint in a callback without setting
    add_cuts and/or add_lazy_constraints to true.

    Attributes:
      events: When the callback should be invoked, by default, never. If an
        unsupported event for a solver/model combination is selected, an
        excecption is raised, see Event above for details.
      mip_solution_filter: restricts the variable values returned in
        CallbackData.solution (the callback argument) at each MIP_SOLUTION event.
        By default, values are returned for all variables.
      mip_node_filter: restricts the variable values returned in
        CallbackData.solution (the callback argument) at each MIP_NODE event. By
        default, values are returned for all variables.
      add_cuts: The callback may add "user cuts" (linear constraints that
        strengthen the LP without cutting of integer points) at MIP_NODE events.
      add_lazy_constraints: The callback may add "lazy constraints" (linear
        constraints that cut off integer solutions) at MIP_NODE or MIP_SOLUTION
        events.
    """
    events: Set[Event] = dataclasses.field(default_factory=set)
    mip_solution_filter: sparse_containers.VariableFilter = sparse_containers.VariableFilter()
    mip_node_filter: sparse_containers.VariableFilter = sparse_containers.VariableFilter()
    add_cuts: bool = False
    add_lazy_constraints: bool = False

    def to_proto(self) -> callback_pb2.CallbackRegistrationProto:
        """Returns an equivalent proto to this CallbackRegistration."""
        result = callback_pb2.CallbackRegistrationProto()
        result.request_registration[:] = sorted([event.value for event in self.events])
        result.mip_solution_filter.CopyFrom(self.mip_solution_filter.to_proto())
        result.mip_node_filter.CopyFrom(self.mip_node_filter.to_proto())
        result.add_cuts = self.add_cuts
        result.add_lazy_constraints = self.add_lazy_constraints
        return result