import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
class WorkflowExecution(SWFBase):
    """An instance of a workflow."""
    workflowId = None
    runId = None

    @wraps(Layer1.signal_workflow_execution)
    def signal(self, signame, **kwargs):
        """SignalWorkflowExecution."""
        self._swf.signal_workflow_execution(self.domain, signame, self.workflowId, **kwargs)

    @wraps(Layer1.terminate_workflow_execution)
    def terminate(self, **kwargs):
        """TerminateWorkflowExecution (p. 103)."""
        return self._swf.terminate_workflow_execution(self.domain, self.workflowId, **kwargs)

    @wraps(Layer1.get_workflow_execution_history)
    def history(self, **kwargs):
        """GetWorkflowExecutionHistory."""
        return self._swf.get_workflow_execution_history(self.domain, self.runId, self.workflowId, **kwargs)['events']

    @wraps(Layer1.describe_workflow_execution)
    def describe(self):
        """DescribeWorkflowExecution."""
        return self._swf.describe_workflow_execution(self.domain, self.runId, self.workflowId)

    @wraps(Layer1.request_cancel_workflow_execution)
    def request_cancel(self):
        """RequestCancelWorkflowExecution."""
        return self._swf.request_cancel_workflow_execution(self.domain, self.workflowId, self.runId)