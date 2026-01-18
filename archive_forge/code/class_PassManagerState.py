from dataclasses import dataclass, field
from enum import Enum
@dataclass
class PassManagerState:
    """A portable container object that pass manager tasks communicate through generator.

    This object can contain every information about the running pass manager workflow,
    except for the IR object being optimized.
    The data structure consists of two elements; one for the status of the
    workflow itself, and another one for the additional information about the IR
    analyzed through pass executions. This container aims at just providing
    a robust interface for the :meth:`.Task.execute`, and no logic that modifies
    the container elements must be implemented.

    This object is mutable, and might be mutated by pass executions.
    """
    workflow_status: WorkflowStatus
    'Status of the current compilation workflow.'
    property_set: PropertySet
    'Information about IR being optimized.'