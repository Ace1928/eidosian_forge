from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowStillActiveError(WorkflowError):

    def __init__(self, operation: str, workflow_id: str):
        self.message = f"{operation} couldn't be completed because Workflow[id={workflow_id}] is still running or pending."
        super().__init__(self.message)