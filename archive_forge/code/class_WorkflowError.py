from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowError(Exception):
    """Workflow error base class."""