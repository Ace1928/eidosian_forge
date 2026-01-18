from ray.util.annotations import PublicAPI
from ray.workflow.common import TaskID
@PublicAPI(stability='alpha')
class WorkflowTaskNotRecoverableError(WorkflowNotResumableError):
    """Raise the exception when we find a workflow task cannot be recovered
    using the checkpointed inputs."""

    def __init__(self, task_id: TaskID):
        self.message = f'Workflow task[id={task_id}] is not recoverable'
        super(WorkflowError, self).__init__(self.message)