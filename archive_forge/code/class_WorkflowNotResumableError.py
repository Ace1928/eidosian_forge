from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowNotResumableError(WorkflowError):
    """Raise the exception when we cannot resume from a workflow."""

    def __init__(self, workflow_id: str):
        self.message = f'Workflow[id={workflow_id}] is not resumable.'
        super().__init__(self.message)