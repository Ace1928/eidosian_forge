from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowNotFoundError(WorkflowError):

    def __init__(self, workflow_id: str):
        self.message = f"Workflow[id={workflow_id}] was referenced but doesn't exist."
        super().__init__(self.message)